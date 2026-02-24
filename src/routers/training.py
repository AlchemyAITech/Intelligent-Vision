from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import asyncio
from src.ultralytics_engine import engine_instance

router = APIRouter()

class TrainRequest(BaseModel):
    project_name: str
    job_id: str
    yaml_path: str
    model_type: str = "yolov8n"
    epochs: int = 10
    batch_size: int = 8
    optimizer: str = "auto"

# 维护大屏连入的监控节点
active_connections = {}

@router.websocket("/ws/{job_id}")
async def websocket_training_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_connections[job_id] = websocket
    print(f">> [Training WS] 终端接入: {job_id}")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        print(f">> [Training WS] 终端断开: {job_id}")
        if job_id in active_connections:
            del active_connections[job_id]

@router.post("/start")
async def start_training(req: TrainRequest):
    ws = active_connections.get(req.job_id)

    # 将算力密集型的大模型训练抛入 asyncio Background Tasks
    asyncio.create_task(
        engine_instance.train_model(
            job_id=req.job_id,
            project_name=req.project_name,
            yaml_path=req.yaml_path,
            model_type=req.model_type,
            epochs=req.epochs,
            batch_size=req.batch_size,
            optimizer=req.optimizer,
            callback_ws=ws
        )
    )

    return {
        "status": "success", 
        "message": f"引擎已加速启动 - Job: {req.job_id} | Device: {engine_instance.device}"
    }

@router.post("/export_onnx/{project_name}/{job_id}")
async def export_to_onnx(project_name: str, job_id: str):
    """跨平台 ONNX 一键导出"""
    try:
        path = engine_instance.export_onnx(project_name, job_id)
        return {"status": "success", "onnx_path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
