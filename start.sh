#!/bin/bash
# start.sh - 一键启动 Intelligent Vision 核心基建 (包含前端紫荆花大盘与训练平台引擎)

# ANSI 颜色码
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${PURPLE}=========================================${NC}"
echo -e "${CYAN}  Intelligent Vision Labs - Core Engine  ${NC}"
echo -e "${PURPLE}=========================================${NC}"

# 强制进入当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 .venv 虚拟环境
if [ -d ".venv" ]; then
    echo -e "${GREEN}>> [OK] 发现确系 Python 虚拟环境 (.venv)，挂载源动力中...${NC}"
    PYTHON_CMD="./.venv/bin/python"
elif [ -d "venv" ]; then
    echo -e "${GREEN}>> [OK] 发现确系 Python 虚拟环境 (venv)，挂载源动力中...${NC}"
    PYTHON_CMD="./venv/bin/python"
else
    echo -e "${YELLOW}>> [WARN] 未探知局部虚拟环境，回退使用系统级全局 python...${NC}"
    PYTHON_CMD="python3"
fi

# 安装大模型底座与可视化依赖
echo -e "${CYAN}>> [依赖检查] 验证 Ultralytics, scikit-learn(降维), FastAPI 等核心库...${NC}"
$PYTHON_CMD -m pip install -q fastapi "uvicorn[standard]" python-multipart websockets ultralytics scikit-learn

echo -e "${GREEN}>> [核心起飞] 正在唤醒 FastAPI 异步后端引擎...${NC}"
echo -e "${PURPLE}>> 🍎 Apple MPS (Metal Performance Shaders) 异构加速集群已在待命状态${NC}"
echo -e "${CYAN}>> 控制台挂载完毕! 请在浏览器尽情访问: http://localhost:8000${NC}"
echo -e "${PURPLE}=========================================${NC}"

# 启动 Uvicorn，将 static 挂载到主站
$PYTHON_CMD -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
