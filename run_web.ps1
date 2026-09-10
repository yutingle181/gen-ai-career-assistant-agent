# 一键启动 Streamlit 演示界面（自动激活 .venv）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path "$root\.venv")) {
    Write-Host "未检测到 .venv，正在创建虚拟环境…" -ForegroundColor Yellow
    python -m venv "$root\.venv"
    & "$root\.venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
}

Write-Host "启动 Streamlit：http://localhost:8501" -ForegroundColor Cyan
& "$root\.venv\Scripts\python.exe" -m streamlit run "$root\app.py" `
    --server.headless true --browser.gatherUsageStats false --server.port 8501
