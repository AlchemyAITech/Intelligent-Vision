import os
import sys
import uuid
import cv2
import base64
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# ====== Import SAM 3 Repo ======
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sum3_repo"))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from sam3.model_builder import build_sam3_image_model
except ImportError as e:
    print(f"Error importing SAM 3: {e}")
    build_sam3_image_model = None

# ====== API Router ======
router = APIRouter()

# ====== 1. Model Initialization, Preprocessing & Inference ======
class SAM3Backend:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.predictor = None
        self.sessions = {}
        self._init_model()

    def _init_model(self):
        print(f"Initializing SAM 3 model on {self.device}...")
        if build_sam3_image_model is None:
            print("SAM 3 dependencies not found. Backend will not work.")
            return

        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "sam3.pt"))
        
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
        if not self.predictor:
            raise ValueError("SAM 3 Predictor is not initialized.")
            
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid Image.")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"Preprocessing image {session_id} - extracting features...")
        self.predictor.set_image(img_rgb)
        
        self.sessions[session_id] = {
            "shape": img_rgb.shape[:2],
            "is_ready": True
        }
        return {"session_id": session_id, "shape": img_rgb.shape[:2], "message": "Preprocessed and features extracted."}

    def predict_mask(self, session_id: str, prompts: dict):
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
        
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=pts_np,
                point_labels=lbs_np,
                box=box,
                multimask_output=True 
            )
            
            if masks is None or len(masks) == 0:
                raise ValueError("No mask generated.")
                
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            if isinstance(best_mask, torch.Tensor):
                best_mask = best_mask.cpu().numpy()
                
            h, w = best_mask.shape
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            mask_bool = best_mask > 0
            bgra[mask_bool] = [185, 128, 41, 153] # Blue-ish translucent overlay
            
            return bgra
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Prediction logic error: {e}")

    def identify_object(self, session_id: str, points: list):
        # In a real enterprise app, we would use CLIP or a similar VL model
        # For this educational lab, we simulated zero-shot logic
        if session_id not in self.sessions:
            raise ValueError("Session not found")
            
        # Return a label based on the context - here we simulate different results for pedagogical variety
        labels = ["智能办公用品", "电子消费产品", "手持移动设备", "实验室试验样本", "视觉识别对象"]
        import random
        # Seed by points to get consistent result for the same object in one session
        if points:
            seed = int(points[0]['x'] + points[0]['y'])
            random.seed(seed)
        
        return random.choice(labels)

# Global instance
sam3_backend = SAM3Backend()

# ====== 2. API Routes ======
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

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    session_id = str(uuid.uuid4())
    try:
        res = sam3_backend.preprocess_image(contents, session_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
async def predict_mask(req: PredictRequest):
    try:
        mask = sam3_backend.predict_mask(req.session_id, req.model_dump())
        _, buffer = cv2.imencode('.png', mask)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return {"mask_base64": f"data:image/png;base64,{b64_str}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/identify")
async def identify_object(req: PredictRequest):
    try:
        label = sam3_backend.identify_object(req.session_id, req.model_dump().get("points", []))
        return {"label": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

