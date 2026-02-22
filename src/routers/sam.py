import os
import sys
import uuid
import cv2
import base64
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- 全局任务状态追踪器 ---
video_task_progress = {}

# ====== Import SAM 3 Repo ======
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sum3_repo"))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Error importing SAM 3: {e}")
    build_sam3_image_model = None
    Sam3Processor = None

try:
    from sam3.model_builder import build_sam3_video_predictor
except ImportError:
    build_sam3_video_predictor = None

# ====== API Router ======
router = APIRouter()

# ====== 1. Model Initialization, Preprocessing & Inference ======
class SAM3Backend:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.predictor = None
        self.image_processor = None
        self.video_predictor = None # 预留给视频预测用的 Predictor
        self.sessions = {}
        self.video_sessions = {}    # 保存视频预测相关的 context 和 inference_state
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
            
            # 初始化包含语言理解头的 Processor
            if Sam3Processor is not None:
                self.image_processor = Sam3Processor(self.model, confidence_threshold=0.3)
                
            print("SAM 3 Image models loaded successfully on CPU.")
            
            # 初始化 Video Predictor
            if build_sam3_video_predictor is not None and model_path and os.path.exists(model_path):
                # 出于性能和显存考虑，实际场景这里可能要共用 Backbone 或者延迟加载。
                # 但根据 notebook，可直接调用 handle_request。这里我们将提供更彻底的占位
                print("Note: Fast tracking implementation.")
                self.video_predictor = build_sam3_video_predictor(checkpoint_path=model_path, device=self.device)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to load SAM 3: {e}")
            self.predictor = None
            self.image_processor = None

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
        
        # 为了兼容 Sam3Processor 文本识别，我们需要提取出 state
        processor_state = {}
        if self.image_processor:
            # 同样塞一份进去给 VL Encoder 计算特征
            import PIL.Image
            pil_image = PIL.Image.fromarray(img_rgb)
            processor_state = self.image_processor.set_image(pil_image)
            
        self.sessions[session_id] = {
            "shape": img_rgb.shape[:2],
            "img_rgb": img_rgb,
            "processor_state": processor_state,
            "is_ready": True
        }
        return {"session_id": session_id, "shape": img_rgb.shape[:2], "message": "Preprocessed and features extracted."}

    def predict_mask(self, session_id: str, prompts: dict):
        if session_id not in self.sessions:
            raise ValueError("Session not found or expired.")
            
        points = prompts.get("points", [])
        boxes = prompts.get("boxes", [])
        text_prompt = prompts.get("text", "")
        
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
        
        # ======== 分支 1：存在文本提示、或者要求全图寻找同类 (走 Sam3Processor) ========
        if (text_prompt or prompts.get("find_similar")) and self.image_processor:
            import torch.nn.functional as F
            try:
                state = self.sessions[session_id]["processor_state"]
                
                # 同步前端阈值到感知引擎
                threshold = prompts.get("similarity_threshold", prompts.get("text_threshold", 0.5))
                self.image_processor.set_confidence_threshold(threshold, state)

                # 1. 重置以往所有遗留提示
                self.image_processor.reset_all_prompts(state)
                
                # 2. 加入文本提示 (如 "苹果", "shoe")
                prompt_to_set = text_prompt.strip() if text_prompt else ""
                if prompt_to_set:
                    state = self.image_processor.set_text_prompt(state=state, prompt=prompt_to_set)
                
                # 3. 如果还有框提示（叠加框）
                if len(boxes) > 0:
                    from sam3.model.box_ops import normalize_bbox, box_xywh_to_cxcywh
                    
                    b = boxes[-1]
                    h, w = self.sessions[session_id]["shape"]
                    
                    bx, by, bw, bh = b['xmin'], b['ymin'], b['xmax'] - b['xmin'], b['ymax'] - b['ymin']
                    box_input_xywh = torch.tensor([bx, by, bw, bh]).view(-1, 4)
                    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
                    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
                    
                    state = self.image_processor.add_geometric_prompt(
                        state=state, box=norm_box_cxcywh, label=True
                    )
                
                # 4. 如果有点提示（叠加点：这对于同类目标选取非常关键）
                if len(point_coords) > 0:
                    for pt, lbl in zip(point_coords, point_labels):
                        # pt_label: 1 is positive, 0 is negative
                        state = self.image_processor.add_point_prompt(
                            state=state, point=pt, label=lbl
                        )
                # 5. 对于纯点面特征的扩散（find_similar 借由多锚点提问），确保推理被触发
                # 在 JIT 版本的 SAM3Processor 中，如果要拿 masks，如果刚才没有传实质文本，往往需要在点后面再调用点什么，或者说 set_text_prompt 设置空字符串已经能够得到 state['masks'] 了。
                # 保险起见，打印个日志确认这里的 state 有没有 mask
                
                out_masks = state.get("masks") # [N, H, W]
                scores = state.get("scores")
                print(f"[SAM3Backend] find_similar scores len: {len(scores) if scores is not None else 0}, text: '{prompt_to_set}'")
                
                if out_masks is None or len(out_masks) == 0:
                    # 对于空返回的自然语言，我们平滑回退，不抛出致命报错
                    return np.zeros((self.sessions[session_id]["shape"][0], self.sessions[session_id]["shape"][1], 4), dtype=np.uint8), []
                
                # [关键修复]：SAM3 返回的 masks 是低分辨特征层 (如 256x256)，必须通过双线性插值拉伸回原始分辨率，否则前端 IOU 数组长度将完全不匹配！
                h_orig, w_orig = self.sessions[session_id]["shape"]
                
                # 确保 out_masks 是 [N, H, W] 的 3D 格式
                if len(out_masks.shape) == 2:
                    out_masks = out_masks.unsqueeze(0) # [1, H, W]
                elif len(out_masks.shape) > 3:
                    out_masks = out_masks.squeeze() # 尝试压平
                    if len(out_masks.shape) == 2:
                        out_masks = out_masks.unsqueeze(0)
                
                # out_masks: [N, H, W] -> 增加 batch 维变身为 [1, N, H, W] (把 N 当做 channel 看待)
                out_masks_4d = out_masks.unsqueeze(0).float()
                
                # 强制按照前台最初传上来的实际原图宽高等比拉出高分辨率 mask
                out_masks_upscaled = F.interpolate(
                    out_masks_4d,
                    size=(h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False
                )
                
                # 恢复为 [N, H_orig, W_orig]
                out_masks_upscaled = out_masks_upscaled.squeeze(0) 
                # 将插值恢复好的高分辨张量覆盖回去
                out_masks = (out_masks_upscaled > 0).float() # 在缩放后重新二值化

                # 提取掩码：如果开启了同类寻找，或者纯文本全图提词，才把符合阈值的全部混入多目标序列
                multi_bgra_list = [] # List of tuples: (mask_np, score_float)
                
                is_pure_text_query = bool(text_prompt and len(point_coords) == 0 and len(boxes) == 0)
                
                if prompts.get("find_similar") or is_pure_text_query:
                    if prompts.get("find_similar"):
                        threshold = prompts.get("similarity_threshold", 0.3)
                    else:
                        threshold = prompts.get("text_threshold", 0.5)
                        
                    valid_indices = torch.where(scores > threshold)[0]
                    
                    if len(valid_indices) == 0:
                        # 兜底：如果连低阈值都没找到，至少返回最有信心的那个
                        best_idx = torch.argmax(scores).item() if scores is not None and len(scores) > 0 else 0
                        best_score = float(scores[best_idx].item()) if scores is not None and len(scores) > 0 else 0.0
                        best_mask = np.squeeze(out_masks[best_idx].cpu().numpy())
                        multi_bgra_list.append((best_mask, best_score))
                    else:
                        # 混合所有过线的目标，并保留多目标副本
                        best_mask = None
                        for idx in valid_indices:
                            score_val = float(scores[idx].item())
                            mask_np = np.squeeze(out_masks[idx].cpu().numpy())
                            multi_bgra_list.append((mask_np, score_val))
                            if best_mask is None:
                                best_mask = mask_np
                            else:
                                best_mask = np.logical_or(best_mask, mask_np)
                else:
                    # [修复文本污染]：如果仅仅是传了文本辅助点/框修正，绝不越权返回群像，只牢牢锁定返回置信度 Top 1
                    best_idx = torch.argmax(scores).item() if scores is not None and len(scores) > 0 else 0
                    best_score = float(scores[best_idx].item()) if scores is not None and len(scores) > 0 else 0.0
                    best_mask = out_masks[best_idx].cpu().numpy()
                    best_mask = np.squeeze(best_mask) # 强制降维防止拆包错误
                
                # 如果降维后还是长宽以上的维度，我们要保底取最后一维的两面
                if len(best_mask.shape) >= 2:
                    h, w = best_mask.shape[-2], best_mask.shape[-1]
                else:
                    h, w = self.sessions[session_id]["shape"]
                    
                color = prompts.get("mask_color", [185, 128, 41, 153]) # [B, G, R, A]
                
                # 总合并图渲染
                bgra = np.zeros((h, w, 4), dtype=np.uint8)
                mask_bool = best_mask > 0
                bgra[mask_bool] = color
                
                # 各子图平行渲染及其分数绑定
                rendered_multi_bgra = []
                for sub_mask, score_val in multi_bgra_list:
                    sub_bgra = np.zeros((h, w, 4), dtype=np.uint8)
                    sub_bool = sub_mask > 0
                    sub_bgra[sub_bool] = color
                    rendered_multi_bgra.append({"mask": sub_bgra, "score": score_val})
                    
                return bgra, rendered_multi_bgra
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise ValueError(f"Processor logic error: {e}")
        
        # ======== 分支 2：原版的单图交互点/框 ========
        else:
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
                
                # Use provided color or default
                color = prompts.get("mask_color", [185, 128, 41, 153]) # [B, G, R, A]
                bgra[mask_bool] = color
                
                return bgra, []
                
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

class UploadRequest(BaseModel):
    image_base64: str = ""
    session_id: str = ""

class PredictRequest(BaseModel):
    session_id: str
    points: List[PointPrompt] = []
    boxes: List[BoxPrompt] = []
    text: str = ""
    mask_color: List[int] = [185, 128, 41, 153]
    v_width: Optional[float] = None
    v_height: Optional[float] = None
    find_similar: bool = False
    similarity_threshold: float = 0.3
    text_threshold: float = 0.4

class VideoStartRequest(BaseModel):
    video_path: str

class VideoPromptRequest(BaseModel):
    session_id: str
    frame_idx: int
    obj_id: int
    points: List[PointPrompt] = []
    boxes: List[BoxPrompt] = []
    text: str = ""
    mask_color: List[int] = [185, 128, 41, 153]
    v_width: Optional[float] = None
    v_height: Optional[float] = None

@router.post("/upload")
async def upload_image(
    file: Optional[UploadFile] = File(None),
    req: Optional[UploadRequest] = None
):
    """
    支持两种模式：
    1. Multipart/form-data (关键：file 字段)
    2. Application/json (关键：image_base64 字段)
    """
    contents = None
    session_id = str(uuid.uuid4())
    
    if file:
        contents = await file.read()
    elif req and req.image_base64:
        import base64
        contents = base64.b64decode(req.image_base64)
        if req.session_id:
            session_id = req.session_id
            
    if not contents:
        raise HTTPException(status_code=400, detail="No image content provided.")
        
    try:
        res = sam3_backend.preprocess_image(contents, session_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
async def predict_mask(req: PredictRequest):
    try:
        mask, multi_masks = sam3_backend.predict_mask(req.session_id, req.model_dump())
        
        # 编码主合并提取图
        _, buffer = cv2.imencode('.png', mask)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        main_mask_b64 = f"data:image/png;base64,{b64_str}"
        
        # 编码各个分离的目标子图
        multi_b64_list = []
        for child_item in multi_masks:
            child_mask = child_item["mask"]
            score_val = child_item["score"]
            _, c_buffer = cv2.imencode('.png', child_mask)
            cb64_str = base64.b64encode(c_buffer).decode('utf-8')
            multi_b64_list.append({
                "mask_base64": f"data:image/png;base64,{cb64_str}",
                "score": score_val
            })
            
        return {
            "mask_base64": main_mask_b64,
            "multi_masks_base64": multi_b64_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/identify")
async def identify(req: PredictRequest):
    try:
        points_list = [{"x": p.x, "y": p.y, "label": p.label} for p in req.points]
        label = sam3_backend.identify_object(req.session_id, points_list)
        return {"label": label, "score": 0.95}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ================= 视频追踪增补 API =================

@router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """保存前端推来的视频并返回其绝对路径."""
    import shutil
    import os
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "video")
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 实际业务这里应转码或提供给 SAM Video 模型
        return {"message": "Video uploaded", "video_path": file_path, "url": f"/uploads/video/{file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/video/start_session")
async def video_start_session(req: VideoStartRequest):
    """根据上传好的视频获得一个 inference_state。"""
    if sam3_backend.video_predictor is None:
        raise HTTPException(status_code=500, detail="SAM3 Video Predictor is NOT initialized on start.")
    try:
        session_id = str(uuid.uuid4())
        res = sam3_backend.video_predictor.start_session(resource_path=req.video_path, session_id=session_id)
        sam3_backend.video_sessions[session_id] = {
            "video_path": req.video_path,
        }
        return {"session_id": res["session_id"], "message": "Video session started."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/video/add_prompt")
async def video_add_prompt(req: VideoPromptRequest):
    """为视频特定帧追加点/框/文字 prompt。"""
    try:
        import cv2
        if req.session_id not in sam3_backend.video_sessions:
            raise ValueError("Session missing.")
        
        session = sam3_backend.video_sessions[req.session_id]
        
        # 优先从前端透传具有 CSS 正缺拉伸的宽高值
        v_width = req.v_width
        v_height = req.v_height
        
        if not v_width or not v_height:
            # 兼容：如果前端没有传分辨率，就采用底层 OpenCV
            cap = cv2.VideoCapture(session["video_path"])
            v_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            v_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()

        # 将前端传入的 Prompt 转为张量并进行 [0..1] 归一化
        p_coords, p_labels = [], []
        if req.points:
            for p in req.points:
                p_coords.append([p.x / v_width, p.y / v_height])
                p_labels.append(p.label)
            
        pts = np.array(p_coords, dtype=np.float32) if p_coords else None
        lbs = np.array(p_labels, dtype=np.int32) if p_labels else None
        
        # 视频跟踪接收的是 normalize 之后的 [xmin, ymin, w, h] 格式
        boxes = None
        box_labels = None
        if len(req.boxes) > 0:
            b = req.boxes[-1]
            bx_nw = (b.xmax - b.xmin) / v_width
            bx_nh = (b.ymax - b.ymin) / v_height
            bx_nx = b.xmin / v_width
            bx_ny = b.ymin / v_height
            boxes = [[bx_nx, bx_ny, bx_nw, bx_nh]]
            box_labels = [req.obj_id] if req.obj_id else [1]
            
        text = req.text if req.text else None

        res = {}
        # 解决模型同时收到框和点互斥冲突的方法：如果存在框，先走一层以框定位目标
        if boxes is not None:
             res = sam3_backend.video_predictor.add_prompt(
                 session_id=req.session_id,
                 frame_idx=req.frame_idx,
                 obj_id=req.obj_id,
                 points=None,
                 point_labels=None,
                 bounding_boxes=boxes,
                 bounding_box_labels=box_labels,
                 text=None
             )
        elif text:
             # 有文字则先行跑一遍文字获取特征实体
             res = sam3_backend.video_predictor.add_prompt(
                 session_id=req.session_id,
                 frame_idx=req.frame_idx,
                 obj_id=req.obj_id,
                 points=None,
                 point_labels=None,
                 bounding_boxes=None,
                 bounding_box_labels=None,
                 text=text
             )
        
        # 此后再紧接着用手工描绘的点实施修正，并将最终包含掩码的 res 存下
        if p_coords and len(p_coords) > 0:
             res = sam3_backend.video_predictor.add_prompt(
                 session_id=req.session_id,
                 frame_idx=req.frame_idx,
                 obj_id=req.obj_id,
                 points=p_coords,
                 point_labels=p_labels,
                 bounding_boxes=None, # 防止其报 Assertion
                 bounding_box_labels=None,
                 text=None
             )
        
        # 解析返回的 Outputs 拿 Mask
        frame_idx = res.get("frame_index", req.frame_idx)
        outputs = res.get("outputs", {})
        
        out_obj_ids = outputs.get("out_obj_ids", [])
        out_binary_masks = outputs.get("out_binary_masks", [])
        
        if len(out_obj_ids) > 0 and len(out_binary_masks) > 0:
            obj_ids_list = list(out_obj_ids)
            if req.obj_id in obj_ids_list:
                idx = obj_ids_list.index(req.obj_id)
            else:
                idx = 0
            
            mask = out_binary_masks[idx]
            if len(mask.shape) == 3: # (1, H, W)
                mask = mask[0]
            
            h, w = mask.shape
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            color = req.mask_color if req.mask_color else [185, 128, 41, 153]
            bgra[mask > 0] = color
            
            _, buffer = cv2.imencode('.png', bgra)
            # base64 return
            import base64
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return {
                "message": f"Successfully added prompt at frame {req.frame_idx}",
                "mask_base64": f"data:image/png;base64,{b64_str}"
            }
        
        return {"message": f"Successfully added prompt at frame {req.frame_idx}", "mask_base64": None}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

def process_video_tracking_task(session_id: str, video_path: str):
    """后台任务：实际执行视频推理并合成。"""
    try:
        import cv2
        import os
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_task_progress[session_id]["totalFrames"] = total_frames
        
        out_name = f"{session_id}_tracked.mp4"
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "tracked")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 前端可能需要 h.264，这里我们先用 mp4v 提供
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        masks_dict = {}
        for out_dict in sam3_backend.video_predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="both",
            start_frame_idx=0,
            max_frame_num_to_track=None
        ):
            if video_task_progress[session_id].get("stop_requested"):
                break
                
            out_frame_idx = out_dict.get("frame_index")
            video_task_progress[session_id]["progress"] = out_frame_idx + 1 # 实时更新进度
            
            outputs = out_dict.get("outputs", {})
            out_obj_ids = outputs.get("out_obj_ids", [])
            out_mask_logits = outputs.get("out_binary_masks", [])
            masks_dict[out_frame_idx] = {}
            for i, obj_id in enumerate(out_obj_ids):
                # Apply > 0 threshold (or just copy since out_binary_masks is already bool)
                mask = (out_mask_logits[i, 0] > 0.0).cpu().numpy().astype(np.uint8) if hasattr(out_mask_logits[i, 0], 'cpu') else (out_mask_logits[i, 0] > 0.0).astype(np.uint8)
                masks_dict[out_frame_idx][obj_id] = mask
        
        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Color palette for different objects
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        while True:
            if video_task_progress[session_id].get("stop_requested"):
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in masks_dict:
                for idx, (obj_id, mask) in enumerate(masks_dict[frame_idx].items()):
                    color = colors[idx % len(colors)]
                    # Create colored mask overlay
                    colored_mask = np.zeros_like(frame)
                    
                    # Ensure mask matches frame dimensions to avoid IndexError
                    if mask.shape != frame.shape[:2]:
                        if mask.shape == (frame.shape[1], frame.shape[0]):
                             mask = mask.T # Transpose WxH to HxW
                        else:
                             mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                             
                    colored_mask[mask > 0] = color
                    # Blend
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
                    
            out_video.write(frame)
            frame_idx += 1
            
        cap.release()
        out_video.release()
        
        if video_task_progress[session_id].get("stop_requested"):
            # 如果是被外部终止的
            video_task_progress[session_id] = {
                "status": "stopped",
                "message": "Task was manually stopped."
            }
            if os.path.exists(out_path):
                os.remove(out_path)
            # Remove session from engine buffer to free memory
            sam3_backend.video_predictor.reset_state(sam3_backend.video_sessions[session_id]["inference_state"])
        else:
            # 发送处理完成信号
            video_task_progress[session_id]["status"] = "done"
            video_task_progress[session_id]["video_url"] = f"/uploads/tracked/{out_name}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        video_task_progress[session_id] = {
            "status": "error",
            "message": str(e)
        }

@router.post("/video/propagate")
async def video_propagate(req: PredictRequest, background_tasks: BackgroundTasks):
    """启动全局推理（后台）。立即向前端返回响应，允许前端通过轮询查进度"""
    try:
        session_id = req.session_id
        if session_id not in sam3_backend.video_sessions:
            raise ValueError("Session missing.")
            
        video_path = sam3_backend.video_sessions[session_id].get("video_path")
        
        # 初始化该 Session 的状态机
        video_task_progress[session_id] = {
            "session_id": session_id,
            "status": "processing",
            "progress": 0,
            "totalFrames": 0,
            "stop_requested": False,
            "created_at": __import__('time').time()
        }
        
        # 将推流沉重计算砸入 FastAPI 的独立线程池
        background_tasks.add_task(process_video_tracking_task, session_id, video_path)
        
        return {"message": "Propagation tracking started in background.", "session_id": session_id}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/video/status/{session_id}")
async def get_video_status(session_id: str):
    """前端轮询专用：获取特定 Session 的后台掩码处理进度"""
    if session_id not in video_task_progress:
        return {"status": "unknown", "message": "No tracking task found for this session."}
    return video_task_progress[session_id]

@router.get("/video/tasks")
async def get_all_tasks():
    """获取所有历史和当前的视频追踪任务."""
    tasks = list(video_task_progress.values())
    # 按照创建时间倒序
    tasks.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return tasks

@router.delete("/video/tasks/{session_id}")
async def delete_or_stop_task(session_id: str):
    """如果正在运行，则发出中止信号；如果是已完成/失败，则清除记录及文件."""
    if session_id not in video_task_progress:
        raise HTTPException(status_code=404, detail="Task not found.")
        
    task = video_task_progress[session_id]
    if task["status"] == "processing":
        # 标记为中止
        video_task_progress[session_id]["stop_requested"] = True
        return {"message": "Stop signal sent."}
    else:
        # 物理清除
        if "video_url" in task:
            # 拼装真实路径
            file_name = task["video_url"].split("/")[-1]
            file_path = os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "tracked", file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 将内存从字典剔除
        del video_task_progress[session_id]
        return {"message": "Task and artifacts deleted successfully."}
