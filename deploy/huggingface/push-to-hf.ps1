<#
.SYNOPSIS
  把本仓库推到 Hugging Face Spaces（docker SDK，用我们的 Dockerfile 直接构建）。

.DESCRIPTION
  为什么需要脚本而不是直接 git push：
  HF Spaces 要求仓库**根目录**有一个带 YAML front-matter 的 README.md ✓✓，
  而我们的根目录 README.md 是项目文档 ✗ —— 两者会冲突 ✓✗。
  所以脚本做三件事：
    1. 从 master 生成一份"干净导出"（排除 .git / .venv / 测试产物 / 压测原始数据）；
    2. 用 deploy/huggingface/README.md **覆盖**根目录那份（HF 元数据 ✓✓）；
    3. 推到 HF 的 git remote ✓✓。

  推完后 HF 会自动 build（Docker SDK ✓✓）—— 日志在 Space 的 Build 页看 ✓✓。

.PARAMETER HfRemote
  HF 的 git remote 地址。形如 https://huggingface.co/spaces/<你的用户名>/<space 名>
  也可以只传 <用户名>/<space 名>，脚本会补全 ✓。

.PARAMETER DryRun
  只生成本地导出目录、不推送（**第一次建议先跑这个** ✓✓ 确认内容对了再推 ✓✓）。

.EXAMPLE
  # 第一次：先看导出内容
  powershell -ExecutionPolicy Bypass -File deploy\huggingface\push-to-hf.ps1 -HfRemote yutingle181/genai-career-assistant -DryRun

.EXAMPLE
  # 确认无误后真正推送
  powershell -ExecutionPolicy Bypass -File deploy\huggingface\push-to-hf.ps1 -HfRemote yutingle181/genai-career-assistant
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HfRemote,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$Repo   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Export = Join-Path $Repo 'Agent_output\hf-space-export'
$HfReadme = Join-Path $PSScriptRoot 'README.md'

if ($HfRemote -notmatch '^https?://') {
    $HfRemote = "https://huggingface.co/spaces/$HfRemote"
}
$HfRemote = $HfRemote.TrimEnd('/') + '.git'

function Ok([string]$m)   { Write-Host "  [OK]   $m" }
function Warn([string]$m) { Write-Host "  [注意] $m" }

Write-Host '=== 0. 前置检查 ==='
if (-not (Test-Path $HfReadme)) { throw "找不到 $HfReadme" }
if (-not (Test-Path (Join-Path $Repo 'Dockerfile'))) { throw '找不到 Dockerfile' }
Ok 'HF 元数据 README 与 Dockerfile 都在'

Write-Host ('  目标 remote = ' + $HfRemote)
Write-Host ('  导出目录   = ' + $Export)

# ---------------------------------------------------------------- 1. 干净导出
Write-Host ''
Write-Host '=== 1. 生成干净导出（git archive，只含被 Git 跟踪的文件 ✓） ==='
if (Test-Path $Export) { Remove-Item $Export -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Export | Out-Null

Push-Location $Repo
try {
    # git archive 只导出**已提交**的内容 ✓✓ —— 天然排除 .venv / __pycache__ / 本地数据 ✓✓
    # 同时也保证推上去的是 master 的真实状态，不会把半成品带上去 ✗✗
    #
    # ⚠️ 必须用 -o 直写文件，**不能管道给 Expand-Archive** ✗✗ ——
    # PowerShell 管道会把二进制流当字符串处理，zip 头（PK\x03\x04）被拆成字符 ✗✗，
    # 报错是 "The path 'PK' either does not exist or is not a valid file system path" ✗。
    # 这个坑和 mysqldump 那次一模一样（DEV_NOTES 记过 ✗✗ 今天又踩一次 ✗✗）。
    $zip = Join-Path $env:TEMP 'hf-space-export.zip'
    if (Test-Path $zip) { Remove-Item $zip -Force }
    git archive --format=zip -o $zip HEAD
    Expand-Archive -Path $zip -DestinationPath $Export -Force
    Remove-Item $zip -Force -EA SilentlyContinue
    Ok ('导出 ' + (Get-ChildItem $Export -Recurse -File).Count + ' 个文件')
} finally {
    Pop-Location
}

# ---------------------------------------------------------------- 2. 覆盖根 README
Write-Host ''
Write-Host '=== 2. 用 HF 版 README 覆盖根目录那份（HF 要求根目录有元数据 ✓） ==='
Copy-Item $HfReadme (Join-Path $Export 'README.md') -Force
Ok 'README.md 已替换为 HF Spaces 元数据版'

# HF 不需要这些（留着只会让构建变慢/泄露信息 ✗✗）
$drop = @('.github', 'deploy', 'docs', 'tests', 'smoke_test.py', 'Agent_output', '.env.example')
foreach ($d in $drop) {
    $p = Join-Path $Export $d
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force -EA SilentlyContinue
        Write-Host ('  已剔除 ' + $d)
    }
}
# .env 绝不能推（含模型 Key ✗✗）—— git archive 不会带它（已被 gitignore ✓✓），双保险再查一次
$envFile = Join-Path $Export '.env'
if (Test-Path $envFile) {
    Remove-Item $envFile -Force
    Warn '导出目录里出现了 .env，已删除（正常不该有 ✓✓ gitignore 挡住了 ✓）'
} else {
    Ok '.env 未被导出 ✓（密钥安全 ✓✓）'
}

Write-Host ''
Write-Host '=== 导出内容预览 ==='
Get-ChildItem $Export | Select-Object -First 20 | ForEach-Object {
    $mark = if ($_.PSIsContainer) { '[DIR] ' } else { '      ' }
    Write-Host ('  ' + $mark + $_.Name)
}
$size = [math]::Round((Get-ChildItem $Export -Recurse -File | Measure-Object -Sum Length).Sum / 1MB, 2)
Write-Host ('  总大小 = ' + $size + ' MB')

# ---------------------------------------------------------------- 3. 推送
if ($DryRun) {
    Write-Host ''
    Write-Host '=== DryRun：不推送 ✓ ==='
    Write-Host ('  请检查 ' + $Export)
    Write-Host '  确认无误后去掉 -DryRun 再跑一次 ✓✓'
    exit 0
}

Write-Host ''
Write-Host '=== 3. 推送到 Hugging Face ==='
Push-Location $Export
try {
    if (-not (Test-Path (Join-Path $Export '.git'))) {
        git init -b main 2>&1 | ForEach-Object { Write-Host ('  ' + $_) }
    }
    git remote remove hf 2>$null
    git remote add hf $HfRemote
    git add -A
    git -c user.email='deploy@local' -c user.name='deploy' commit -m "deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>&1 |
        Select-Object -Last 2 | ForEach-Object { Write-Host ('  ' + $_) }
    git push hf main --force 2>&1 | Select-Object -Last 3 | ForEach-Object { Write-Host ('  ' + $_) }
    Ok '已推送，HF 会自动开始 build（Docker SDK ✓✓）'
} finally {
    Pop-Location
}

Write-Host ''
Write-Host '=== 接下来 ==='
Write-Host '  1. 打开 Space 页面看 Build 日志 ✓✓（首次约 3-6 分钟 ✓ 要下载本地 BGE 模型 ✓✓）'
Write-Host '  2. Build 成功后先点开 /healthz（触发冷启动 ✓✓ 免得上手时干等 ✗✗）'
Write-Host '  3. 去 Settings → Variables and secrets 配 Key（见 deploy/huggingface/README.md 的表格 ✓✓）'
Write-Host '  4. ⚠️ 48 小时不访问会 sleep ✗✗ —— 面试当天提前几分钟点开 ✓✓'