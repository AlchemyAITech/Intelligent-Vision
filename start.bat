@echo off
chcp 65001 >nul
:: start.bat - 一键启动 Intelligent Vision 核心基建 (包含前端紫荆花大盘与训练平台引擎)

echo =========================================
echo   Intelligent Vision Labs - Core Engine  
echo =========================================

:: 强制进入当前脚本所在目录
cd /d "%~dp0"

:: 检查 .venv 虚拟环境
if exist ".venv" (
    echo ^>^> [OK] 发现确系 Python 虚拟环境 (.venv)，挂载源动力中...
    set PYTHON_CMD=.\.venv\Scripts\python.exe
) else if exist "venv" (
    echo ^>^> [OK] 发现确系 Python 虚拟环境 (venv)，挂载源动力中...
    set PYTHON_CMD=.\venv\Scripts\python.exe
) else (
    echo ^>^> [WARN] 未探知局部虚拟环境，回退使用系统级全局 python...
    set PYTHON_CMD=python
)

:: 安装大模型底座与可视化依赖
echo ^>^> [依赖检查] 验证 Ultralytics, scikit-learn(降维), FastAPI 等核心库...
%PYTHON_CMD% -m pip install -q fastapi "uvicorn[standard]" python-multipart websockets ultralytics scikit-learn

echo ^>^> [核心起飞] 正在唤醒 FastAPI 异步后端引擎...
echo ^>^> 🍎 Apple MPS / CUDA 加速集群已在待命状态
echo ^>^> 控制台挂载完毕! 请在浏览器尽情访问: http://localhost:8000
echo =========================================

:: 自动清理端口 8000 遗留进程
echo ^>^> [环境清理] 正在检测并释放 8000 端口...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr "8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: 启动 Uvicorn，将 static 挂载到主站
%PYTHON_CMD% -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
