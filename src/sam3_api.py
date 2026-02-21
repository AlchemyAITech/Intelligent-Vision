import os
import sys
import uuid
import cv2
import base64
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

# ====== 导入 SAM 3 仓库 ======
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sum3_repo"))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from sam3.model_builder import build_sam3_image_model
except ImportError as e:
    print(f"Error importing SAM 3: {e}")
    build_sam3_image_model = None

# ====== FastAPI 初始化 ======
app = FastAPI(title="SAM 3 Interactive API (CPU)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 1. 模型初始化、预处理与交互类隔离 ======
class SAM3Backend:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.predictor = None
        self.sessions = {}
        # 1.1 模型初始化 (Model Initialization)
        self._init_model()

    def _init_model(self):
        print(f"Initializing SAM 3 model on {self.device}...")
        if build_sam3_image_model is None:
            print("SAM 3 dependencies not found. Backend will not work.")
            return

        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "sam3.pt"))
        
        # 为了防止本地没有模型导致崩溃，我们允许 load_from_HF
        try:
            self.model = build_sam3_image_model(
                checkpoint_path=model_path if os.path.exists(model_path) else None,
                device=self.device,
                enable_inst_interactivity=True,
                load_from_HF=(not os.path.exists(model_path))
            )
            self.predictor = self.model.inst_interactive_predictor
            # Patch the interactive predictor's tracker model to use the backbone
            if hasattr(self.predictor.model, 'backbone') and self.predictor.model.backbone is None:
                self.predictor.model.backbone = self.model.backbone
            print("SAM 3 loaded successfully on CPU.")
        except Exception as e:
            print(f"Failed to load SAM 3: {e}")
            self.predictor = None

    def preprocess_image(self, image_bytes: bytes, session_id: str):
        """ 1.2 图像预处理 (Image Preprocessing) """
        if not self.predictor:
            raise ValueError("SAM 3 Predictor is not initialized.")
            
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid Image.")
            
        # SAM 需要 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"Preprocessing image {session_id} - extracting features...")
        # 提取特征，计算 embedding 耗时较长（尤其是在 CPU 上）
        self.predictor.set_image(img_rgb)
        
        self.sessions[session_id] = {
            "shape": img_rgb.shape[:2],
            "is_ready": True
        }
        return {"session_id": session_id, "shape": img_rgb.shape[:2], "message": "Preprocessed and features extracted."}

    def predict_mask(self, session_id: str, prompts: dict):
        """ 1.3 前端交互 (Frontend Interaction - Inference) """
        if session_id not in self.sessions:
            raise ValueError("Session not found or expired.")
            
        points = prompts.get("points", [])
        boxes = prompts.get("boxes", [])
        
        point_coords = []
        point_labels = []
        for pt in points:
            point_coords.append([pt['x'], pt['y']])
            point_labels.append(pt['label'])
            
        box = None
        if len(boxes) > 0:
            b = boxes[-1]
            box = np.array([b['xmin'], b['ymin'], b['xmax'], b['ymax']])
            
        pts_np = np.array(point_coords, dtype=np.float32) if len(point_coords) > 0 else None
        lbs_np = np.array(point_labels, dtype=np.int32) if len(point_labels) > 0 else None
        
        print(f"Predicting mask for session {session_id} with {len(points)} points and {len(boxes)} boxes.")
        
        try:
            # SAM 3 推理出 mask
            masks, scores, logits = self.predictor.predict(
                point_coords=pts_np,
                point_labels=lbs_np,
                box=box,
                multimask_output=True # 取多重中最好的，或是直接用第一层
            )
            
            if masks is None or len(masks) == 0:
                raise ValueError("No mask generated.")
                
            # 多输出情况下，取最高置信度的 mask
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            if isinstance(best_mask, torch.Tensor):
                best_mask = best_mask.cpu().numpy()
                
            # 创建带有透明通道的 BGRA 图像
            h, w = best_mask.shape
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            
            # 使用类似于 "rgba(41, 128, 185, 0.6)" 的颜色
            # OpenCV 使用 BGRA 顺序，所以 Blue=185, Green=128, Red=41, Alpha=153 (约 60%)
            mask_bool = best_mask > 0
            bgra[mask_bool] = [185, 128, 41, 153]
            
            return bgra
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Prediction logic error: {e}")

# 初始化 Backend
sam3_backend = SAM3Backend()

# ====== 2. API 路由定义 ======
class PointPrompt(BaseModel):
    x: float
    y: float
    label: int

class BoxPrompt(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class PredictRequest(BaseModel):
    session_id: str
    points: List[PointPrompt] = []
    boxes: List[BoxPrompt] = []

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    session_id = str(uuid.uuid4())
    try:
        res = sam3_backend.preprocess_image(contents, session_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict_mask(req: PredictRequest):
    try:
        mask = sam3_backend.predict_mask(req.session_id, req.model_dump())
        # 将生成的 mask 转成 base64 形式，便于前端直接作为透明图层覆盖
        _, buffer = cv2.imencode('.png', mask)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return {"mask_base64": f"data:image/png;base64,{b64_str}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 允许局域网访问
    uvicorn.run(app, host="0.0.0.0", port=8000)
