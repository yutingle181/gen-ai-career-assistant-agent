<#
.SYNOPSIS
  启动本机 Redis（Agent 的任务状态后端），幂等：已在跑就直接跳过。

.DESCRIPTION
  为什么需要这个脚本：
  1. `redis-server --service-install` 要管理员权限，普通开发机往往没有，
     所以这里改成"普通用户可跑的启动脚本"，而不是强推 Windows 服务；
  2. 一定要带 `--save ""`：Redis 的 `dir ./` 是相对路径，若当前工作目录不可写，
     后台 RDB 保存会失败，而 `stop-writes-on-bgsave-error` 会把**所有写命令**全禁掉
     （报 MISCONF ... not able to persist on disk），表现为"Redis 在跑但一律写失败"。
  3. 幂等：端口已在监听就跳过，避免重复起进程抢端口。

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_redis.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 6379,
    [string]$RedisHome = 'D:\env\redis',
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

function Write-Ok([string]$m) { Write-Host "  [OK]   $m" }
function Write-Warn2([string]$m) { Write-Host "  [注意] $m" }
function Write-Err([string]$m) { Write-Host "  [失败] $m" }

$server = Join-Path $RedisHome 'redis-server.exe'
$cli = Join-Path $RedisHome 'redis-cli.exe'
$conf = Join-Path $RedisHome 'redis.windows.conf'

Write-Host '=== Redis 启动检查 ==='

if (-not (Test-Path $server)) {
    Write-Err "找不到 $server"
    Write-Host "  可用 -RedisHome <目录> 指定 redis-server.exe 所在目录；" +
    "若尚未安装，Windows 版可用 Memurai，或用 WSL2 起 redis。"
    exit 1
}

# 幂等：端口已在监听就直接放行
if (Get-NetTCPConnection -LocalPort $Port -State Listen -EA SilentlyContinue) {
    $pong = (& $cli -p $Port ping 2>$null | Select-Object -First 1)
    if ($pong -eq 'PONG') {
        Write-Ok "Redis 已在 :$Port 运行"
        exit 0
    }
    Write-Warn2 ":$Port 已被占用但 ping 不通，请先停掉占用进程"
    exit 1
}

$argList = @()
if (Test-Path $conf) { $argList += $conf }
# dir 用绝对路径 + 关掉 RDB：避免上面说的 MISCONF 把写命令全禁掉
$argList += @('--port', "$Port", '--dir', $RedisHome, '--save', '""', '--appendonly', 'no')

Write-Host "  启动：redis-server $($argList -join ' ')"
$proc = Start-Process -FilePath $server -ArgumentList $argList -WindowStyle Hidden -PassThru

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    $pong = (& $cli -p $Port ping 2>$null | Select-Object -First 1)
    if ($pong -eq 'PONG') {
        Write-Ok "Redis 已就绪（PID $($proc.Id)，:$(($proc.Id))）"
        # 写权限自检：MISCONF 时 ping 仍是 PONG，只有写才会暴露问题
        $set = (& $cli -p $Port set agent:startup:probe 1 2>$null | Select-Object -First 1)
        if ($set -eq 'OK') {
            & $cli -p $Port del agent:startup:probe 2>$null | Out-Null
            Write-Ok '写命令可用'
        } else {
            Write-Warn2 'ping 通但写失败 —— 多半是 RDB 目录不可写导致的 MISCONF，检查 --dir'
        }
        exit 0
    }
}

Write-Err "启动超时（$TimeoutSeconds 秒），进程 PID $($proc.Id) 仍在但未就绪"
exit 1