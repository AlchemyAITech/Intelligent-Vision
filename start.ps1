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
    Write-Host ">> [OK] 发现确系 Python 虚拟环境 (.venv)，挂载源动力中..." -ForegroundColor Green
    $PYTHON_CMD = ".\.venv\Scripts\python.exe"
}
elseif (Test-Path -Path "venv") {
    Write-Host ">> [OK] 发现确系 Python 虚拟环境 (venv)，挂载源动力中..." -ForegroundColor Green
    $PYTHON_CMD = ".\venv\Scripts\python.exe"
}
else {
    Write-Host ">> [WARN] 未探知局部虚拟环境，回退使用系统级全局 python..." -ForegroundColor Yellow
}

# 安装大模型底座与可视化依赖
Write-Host ">> [依赖检查] 验证 Ultralytics, scikit-learn(降维), FastAPI 等核心库..." -ForegroundColor Cyan
& $PYTHON_CMD -m pip install -q fastapi "uvicorn[standard]" python-multipart websockets ultralytics scikit-learn

Write-Host ">> [核心起飞] 正在唤醒 FastAPI 异步后端引擎..." -ForegroundColor Green
Write-Host ">> 🍎 Apple MPS / CUDA 加速集群已在待命状态" -ForegroundColor Magenta
Write-Host ">> 控制台挂载完毕! 请在浏览器尽情访问: http://localhost:8000" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Magenta

# 自动清理端口 8000 遗留进程
Write-Host ">> [环境清理] 正在检测并释放 8000 端口..." -ForegroundColor Yellow
$port8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($port8000) {
    Stop-Process -Id $port8000.OwningProcess -Force -ErrorAction SilentlyContinue
}

# 启动 Uvicorn，将 static 挂载到主站
& $PYTHON_CMD -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
