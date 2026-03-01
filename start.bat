@echo off
chcp 65001 >nul
:: start.bat - One-click start Intelligent Vision core (frontend + training engine)

echo =========================================
echo   Intelligent Vision Labs - Core Engine  
echo =========================================

:: Change to script directory
cd /d "%~dp0"

:: Check .venv / venv
if exist ".venv" (
    echo ^>^> [OK] Found .venv, activating...
    set PYTHON_CMD=.\.venv\Scripts\python.exe
) else if exist "venv" (
    echo ^>^> [OK] Found venv, activating...
    set PYTHON_CMD=.\venv\Scripts\python.exe
) else (
    echo ^>^> [WARN] No local venv, using system python...
    set PYTHON_CMD=python
)

:: Install deps
echo ^>^> [Deps] Checking Ultralytics, scikit-learn, FastAPI...
%PYTHON_CMD% -m pip install -q fastapi "uvicorn[standard]" python-multipart websockets ultralytics scikit-learn

echo ^>^> [Core] Starting FastAPI backend...
echo ^>^> Apple MPS / CUDA ready
echo ^>^> Open in browser: http://localhost:8000
echo =========================================

:: Clean port 8000
echo ^>^> [Env Cleanup] Checking and releasing port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr "8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: Start Uvicorn
%PYTHON_CMD% -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
