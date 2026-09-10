# 一键启动 FastAPI 服务（自动激活 .venv）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path "$root\.venv")) {
    Write-Host "未检测到 .venv，正在创建虚拟环境…" -ForegroundColor Yellow
    python -m venv "$root\.venv"
    & "$root\.venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
}

Write-Host "启动 FastAPI：http://localhost:8000  文档：http://localhost:8000/docs" -ForegroundColor Cyan
& "$root\.venv\Scripts\python.exe" -m uvicorn src.api.main:app `
    --host 0.0.0.0 --port 8000 --reload
