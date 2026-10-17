<#
.SYNOPSIS
  阶梯并发压测：依次用 1 / 10 / 50 并发压同一组场景，产出 CSV 与环境快照。

.DESCRIPTION
  为什么用阶梯而不是一把压到最高并发：
  压测的价值在于**找到拐点**，而拐点只能从"逐步加压"里看出来。
  一次压到 500 并发只会得到一个"超时了很多"的数字 ✗ —— 既不知道拐点在哪，
  也不知道是 CPU、内存还是某个外部依赖先扛不住 ✓。

  每档都会写一份 CSV（含逐请求耗时），跑完用 bench/plot_bench.py 画曲线 ✓。

  诚实说明：
  - 默认**不压** /chat/stream（它会真调 LLM、烧 token）✓，需要时加 -IncludeChat ✓；
  - baseline（GET /）的 QPS 很高但意义有限：它只证明"框架不是瓶颈" ✓，
    真实路径里最重的是本地 embedding 与 LLM ✓ —— 所以检索场景才是重点 ✓；
  - 本机单进程跑，**不要**把这里的数字当线上容量结论 ✓（单进程 + 本机 CPU ✓）。

.PARAMETER BaseUrl
  被压服务地址。默认 http://127.0.0.1:8000（压测前需自行起服务 ✓）。

.PARAMETER Stages
  阶梯并发列表，默认 "1,10,50"。

.PARAMETER Duration
  每档持续秒数，默认 30。

.PARAMETER OutDir
  输出目录，默认 bench/results（已在 .gitignore 内 ✓）。

.PARAMETER IncludeChat
  是否包含 SSE 首事件场景（**会调用 LLM、消耗 token** ✓）。

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File bench\run_bench.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File bench\run_bench.ps1 -Stages 1,10,50,100 -Duration 60
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [string]$Stages = '1,10,50',
    [int]$Duration = 30,
    [string]$OutDir = '',
    [switch]$IncludeChat
)

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
if (-not $OutDir) { $OutDir = Join-Path $repo 'bench\results' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$python = Join-Path $repo '.venv-ci\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host "  [失败] 找不到 $python —— 请用 .venv-ci 环境运行"
    exit 1
}

Write-Host '=== 0. 环境快照（没有它，压测数字不可复现）==='
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$mem = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$env_lines = @(
    "timestamp=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "python=$(& $python -c 'import platform;print(platform.python_version())')",
    "locust=$(& $python -c 'import locust;print(locust.__version__)')",
    "host=$BaseUrl",
    "cpu=$($cpu.Name.Trim()) cores=$($cpu.NumberOfLogicalProcessors)",
    "mem_gb=$mem",
    "os=$([System.Environment]::OSVersion.VersionString)",
    "stages=$Stages duration_s=$Duration include_chat=$IncludeChat"
)
$envFile = Join-Path $OutDir 'env.txt'
$env_lines | Set-Content -Path $envFile -Encoding UTF8
$env_lines | ForEach-Object { Write-Host ('  ' + $_) }

Write-Host ''
Write-Host '=== 1. 确认被压服务活着 ==='
try {
    $r = Invoke-WebRequest "$BaseUrl/" -UseBasicParsing -TimeoutSec 10
    Write-Host ("  [OK]   $BaseUrl/ 返回 " + $r.StatusCode)
} catch {
    Write-Host ("  [失败] 连不上 $BaseUrl —— 请先起服务：" + $_.Exception.Message)
    exit 1
}

# 预热是**必需**的，不是可选项：
# 实测首次检索要 41.9 秒（要加载本地 BGE 模型 + faiss 索引），第二次起才降到 ~100ms。
# 不预热就直接压，第一个请求会把整个阶段的时延分布彻底带偏 —— 得到的不是系统性能，
# 而是"模型加载有多慢"。同理 GET / 也要先跑一次让中间件与 JIT 热起来。
Write-Host ''
Write-Host '=== 2. 预热（冷启动会让时延分布失真）==='
$warmBody = @{ query = 'warmup'; top_k = 3 } | ConvertTo-Json
$kb = if ($env:BENCH_KB) { $env:BENCH_KB } else { 'samples' }
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $null = Invoke-WebRequest "$BaseUrl/knowledge/$kb/search" -Method POST `
        -ContentType 'application/json' -Body $warmBody -UseBasicParsing -TimeoutSec 120
    $sw.Stop()
    Write-Host ("  首次检索（含冷启动）：" + $sw.ElapsedMilliseconds + "ms  ← 这个数字本身也是结论")
} catch {
    Write-Host ("  [注意] 预热检索失败（检索场景将被跳过）：" + $_.Exception.Message)
}
$null = Invoke-WebRequest "$BaseUrl/" -UseBasicParsing -TimeoutSec 10
Write-Host '  GET / 预热完成'

# 每个场景**单独**跑一轮，而不是混在一起。
# 两个原因（都是实测踩出来的）：
#   1) 混合场景会互相抢 CPU —— 检索要跑本地 embedding，是 CPU 密集的，
#      和 baseline 混在一轮里会把 baseline 的时延也拉高，得到的数据不干净；
#   2) `-u 1` 时 locust 只派**一个** User 类（按权重随机选），
#      于是"1 并发"那一档可能整段都只有 baseline，检索行直接缺失。
$scenarios = if ($IncludeChat) { @('baseline', 'retrieval', 'llm') } else { @('baseline', 'retrieval') }
$stageList = $Stages -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }

foreach ($sc in $scenarios) {
    Write-Host ''
    Write-Host ("########## 场景：" + $sc + " ##########")
    foreach ($s in $stageList) {
        Write-Host ''
        Write-Host "=== $sc · $s 并发 × $Duration 秒 ==="
        $csv = Join-Path $OutDir "$sc-$s.csv"
        # 只跑本轮目标场景：靠 locustfile 里的 BENCH_SCENARIO 环境变量把其余类的 weight 置 0
        # （locust 不会为权重 0 的类建用户 ✓，这是实测唯一有效的"排除"方式 ✓）。
        # 不用 --tags / --class-picker：实测这两个参数在 2.46 上都会把类过滤成空。
        $env:BENCH_SCENARIO = $sc
        $args = @(
            '-m', 'locust',
            '-f', 'bench/locustfile.py',
            '--host', $BaseUrl,
            # --headless 不能省：不加时 locust 会尝试起 Web UI 并**阻塞**，
            # 脚本会挂在那里直到被外部取消（实测踩过）。
            '--headless',
            '-u', $s, '-r', [Math]::Max(1, [int]($s / 2)),
            '-t', "$($Duration)s",
            '--csv', $csv,
            '--only-summary',
            '--loglevel', 'WARNING'
        )
        & $python @args 2>&1 | Select-Object -Last 12 | ForEach-Object { Write-Host ('  ' + $_) }
        $env:BENCH_SCENARIO = $null
        if (Test-Path "$($csv)_stats.csv") {
            Write-Host ("  产物：" + (Split-Path "$($csv)_stats.csv" -Leaf))
        }
    }
}

Write-Host ''
Write-Host '=== 完成，画曲线 ==='
Write-Host ("  .venv-ci\Scripts\python.exe bench\plot_bench.py --results-dir `"$OutDir`"")