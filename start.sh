#!/bin/bash
# start.sh - 一键启动 Intelligent Vision 后端与前端

echo "========================================="
echo "  Intelligent Vision Labs - 启动脚本"
echo "========================================="

# 强制进入当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 .venv 虚拟环境
if [ -d ".venv" ]; then
    echo ">> 发现本地 .venv 环境，准备挂载..."
    PYTHON_CMD="./.venv/bin/python"
elif [ -d "venv" ]; then
    echo ">> 发现本地 venv 环境，准备挂载..."
    PYTHON_CMD="./venv/bin/python"
else
    echo ">> 未发现虚拟环境，回退到全局 python..."
    PYTHON_CMD="python3"
fi

# 安装基础 Web 依赖 (如尚未安装)
echo ">> 检查 FastAPI 等 Web 依赖..."
$PYTHON_CMD -m pip install -q fastapi "uvicorn[standard]" python-multipart websockets

echo ">> 正在启动 FastAPI 后端服务..."
echo ">> (如由于端口冲突导致失败，请手动结束占用 8000 端口的进程)"
echo ">> 应用将在 http://localhost:8000 运行"
echo "========================================="

# 启动 Uvicorn，将 static 挂载到主站
$PYTHON_CMD -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
