<#
.SYNOPSIS
start.ps1 - 一键启动 Intelligent Vision 核心基建 (包含前端紫荆花大盘与训练平台引擎)
#>

$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "  Intelligent Vision Labs - Core Engine  " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Magenta

# 强制进入当前脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $ScriptDir

# 检查 .venv 或者 venv 虚拟环境
$PYTHON_CMD = "python"
if (Test-Path -Path ".venv") {
    Write-Host ">> [OK] Found .venv, activating..." -ForegroundColor Green
    $PYTHON_CMD = ".\.venv\Scripts\python.exe"
}
elseif (Test-Path -Path "venv") {
    Write-Host ">> [OK] Found venv, activating..." -ForegroundColor Green
    $PYTHON_CMD = ".\venv\Scripts\python.exe"
}
else {
    Write-Host ">> [WARN] No local venv found, using system python..." -ForegroundColor Yellow
}

# 安装大模型底座与可视化依赖
Write-Host ">> [Deps] Checking Ultralytics, scikit-learn, FastAPI..." -ForegroundColor Cyan
& $PYTHON_CMD -m pip install -q fastapi "uvicorn[standard]" python-multipart websockets ultralytics scikit-learn

Write-Host ">> [Core] Starting FastAPI backend..." -ForegroundColor Green
Write-Host ">> Apple MPS / CUDA ready" -ForegroundColor Magenta
Write-Host ">> Open in browser: http://localhost:32100" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Magenta

# Auto cleanup process on port 32100
Write-Host ">> [Env Cleanup] Checking and releasing port 32100..." -ForegroundColor Yellow
$port32100 = Get-NetTCPConnection -LocalPort 32100 -State Listen -ErrorAction SilentlyContinue
if ($port32100) {
    Stop-Process -Id $port32100.OwningProcess -Force -ErrorAction SilentlyContinue
}

# 启动 Uvicorn，将 static 挂载到主站
& $PYTHON_CMD -m uvicorn src.main:app --host 0.0.0.0 --port 32100 --reload
