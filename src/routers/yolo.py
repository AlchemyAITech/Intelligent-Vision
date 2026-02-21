import os
import cv2
import sys
from ultralytics import YOLO, SAM
import numpy as np

class YOLOManager:
    """
    Manages YOLO model loading, inference, and result visualization.
    """
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self._model_cache = {}
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self, model_name):
        """
        Loads the model using an internal dict cache.
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]
            
        model_path = os.path.join(self.model_dir, model_name)
        try:
            if os.path.exists(model_path):
                 model = SAM(model_path) if "sam" in model_name.lower() else YOLO(model_path)
            else:
                 root_path = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
                 if os.path.exists(root_path):
                      import shutil
                      shutil.move(root_path, model_path)
                 else:
                      try:
                          _ = SAM(model_name) if "sam" in model_name.lower() else YOLO(model_name)
                      except Exception:
                          pass
                      if os.path.exists(root_path):
                          import shutil
                          shutil.move(root_path, model_path)
                 model = SAM(model_path) if "sam" in model_name.lower() else YOLO(model_path)
            
            self._model_cache[model_name] = model
            return model
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return None

    def predict(self, model, source, conf=0.25, iou=0.45, classes=None, prompts=None):
        """
        Runs prediction on the source (image/frame).
        """
        if model is None: return None
        
        # Check if it's a SAM model (Strict check)
        # OLD: "sam" in str(model.model) ... matches "Upsample" layer!
        is_sam = model.__class__.__name__ == 'SAM'
        
        if is_sam:
            # Prepare args for SAM
            kwargs = {'source': source}
            if prompts:
                if 'bboxes' in prompts: kwargs['bboxes'] = prompts['bboxes']
                if 'points' in prompts: kwargs['points'] = prompts['points']
                if 'labels' in prompts: kwargs['labels'] = prompts['labels']
            
            # SAM 2 / 3 inference
            results = model(**kwargs)
            return results
        else:
            # YOLO inference
            results = model.predict(source, conf=conf, iou=iou, classes=classes, verbose=False)
            
            # Force Manual Filter (Fix for model ignoring args)
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                res = results[0]
                mask = None
                
                # 1. Conf Filter
                if conf is not None:
                     conf_mask = res.boxes.conf >= conf
                     mask = conf_mask
                
                # 2. Class Filter
                if classes is not None:
                     import torch
                     cls_tensor = res.boxes.cls
                     c_mask = torch.zeros_like(cls_tensor, dtype=torch.bool)
                     for c in classes:
                         c_mask |= (cls_tensor == c)
                     
                     if mask is None: mask = c_mask
                     else: mask &= c_mask
                
                if mask is not None:
                     res.boxes = res.boxes[mask]
            
            return results

    def track(self, model, source, conf=0.25, iou=0.45, classes=None, persist=True, prompts=None, tracker="bytetrack.yaml"):
        """
        Runs tracking (for video/stream).
        """
        if model is None: return None
        
        is_sam = model.__class__.__name__ == 'SAM'
        
        if is_sam:
            try:
                # kwargs setup
                kwargs = {'source': source, 'persist': persist}
                # Tracker argument might not be supported by SAM directly in same way
                results = model.track(**kwargs)
                return results
            except Exception as e:
                # Fallback to per-frame predict
                return self.predict(model, source, conf, iou, classes, prompts)
        else:
            # YOLO Tracking
            results = model.track(source, persist=persist, conf=conf, iou=iou, classes=classes, verbose=False, tracker=tracker)
            
            # Force Manual Filter (Tracking)
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                res = results[0]
                mask = None
                if conf is not None:
                     mask = res.boxes.conf >= conf
                if classes is not None:
                     import torch
                     cls_tensor = res.boxes.cls
                     c_mask = torch.zeros_like(cls_tensor, dtype=torch.bool)
                     for c in classes:
                         c_mask |= (cls_tensor == c)
                     if mask is None: mask = c_mask
                     else: mask &= c_mask
                
                if mask is not None:
                     res.boxes = res.boxes[mask]
            
            return results

    def plot_result(self, result, show_conf=True, show_labels=True, show_boxes=True, show_masks=True, show_contours=False, show_pose=True, track_history=None):
        """
        Plots the result and returns an RGB image.
        Unified ID-based coloring for Tracking (Box + Mask + Trail).
        Class-based coloring for Static Detection.
        """
        try:
            import cv2
            import numpy as np
            import torch
            from ultralytics.utils.plotting import colors

            is_tracking = track_history is not None and hasattr(result, 'boxes') and result.boxes is not None and result.boxes.id is not None
            
            # --- 1. Static Mode ---
            if not is_tracking:
                 res_plotted = result.plot(
                     conf=show_conf, 
                     labels=show_labels, 
                     boxes=show_boxes, 
                     masks=show_masks,
                     kpt_line=show_pose,
                     kpt_radius=5 if show_pose else 0
                 )
                 
                 # 掩码高亮边框支持
                 if show_contours and hasattr(result, 'masks') and result.masks is not None:
                      masks_data = result.masks.data.cpu().numpy()
                      cls_ids = result.boxes.cls.int().cpu().tolist() if hasattr(result, 'boxes') and result.boxes is not None else []
                      for i, mask in enumerate(masks_data):
                           c_index = int(cls_ids[i]) if i < len(cls_ids) else 0
                           color = tuple(int(x) for x in colors(c_index, bgr=True))
                           
                           m_bin = (mask > 0.5).astype(np.uint8) * 255
                           cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                           for cnt in cnts:
                                if cv2.contourArea(cnt) < 10: continue
                                cv2.polylines(res_plotted, [cnt], True, color, 2, lineType=cv2.LINE_AA)
                 
                 return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB) if res_plotted is not None else None

            # --- 2. Tracking Mode (ID-based, Unified) ---
            res_plotted = result.plot(
                conf=False, 
                labels=False, 
                boxes=False, 
                masks=False, 
                probs=False,
                kpt_line=show_pose,
                kpt_radius=5 if show_pose else 0
            )
            
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.int().cpu().tolist()
            conf_scores = result.boxes.conf.float().cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            
            masks_data = None
            if hasattr(result, 'masks') and result.masks is not None:
                 masks_data = result.masks.data.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                 id_color = tuple(int(x) for x in colors(track_id, True))
                 
                 # Trail
                 center = (float((boxes_xyxy[i][0] + boxes_xyxy[i][2]) / 2), float((boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2))
                 track = track_history.get(track_id, [])
                 track.append(center)
                 if len(track) > 30: track.pop(0)
                 track_history[track_id] = track
                 
                 points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                 if len(points) > 1:
                      for j in range(len(points) - 1):
                           thickness = int(np.sqrt(float(j + 1)) * 1.5)
                           cv2.line(res_plotted, tuple(points[j][0]), tuple(points[j+1][0]), id_color, thickness)
                 
                 # Mask
                 if show_masks and masks_data is not None and i < len(masks_data):
                      m_bin = (masks_data[i] > 0.5).astype(np.uint8) * 255
                      mask_bool = (m_bin == 255)
                      res_plotted[mask_bool] = (res_plotted[mask_bool] * 0.6 + np.array(id_color) * 0.4).astype(np.uint8)

                 # Contours
                 if (show_masks or show_contours) and masks_data is not None and i < len(masks_data):
                      m_bin = (masks_data[i] > 0.5).astype(np.uint8) * 255
                      cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                      for cnt in cnts:
                           if cv2.contourArea(cnt) < 10: continue
                           cv2.polylines(res_plotted, [cnt], True, id_color, max(1, 2 if show_contours else 1), lineType=cv2.LINE_AA)

                 # Box & Label
                 x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                 if show_boxes:
                      cv2.rectangle(res_plotted, (x1, y1), (x2, y2), id_color, 2)
                 if show_labels:
                      cls_idx = cls_ids[i]
                      label = result.names[cls_idx]
                      text = f"{label} ID:{track_id}"
                      if show_conf: text += f" {conf_scores[i]:.2f}"
                      (w_txt, h_txt), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                      cv2.rectangle(res_plotted, (x1, y1 - 20), (x1 + w_txt, y1), id_color, -1)
                      cv2.putText(res_plotted, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Plot Error: {e}")
            return None

    # --- Solutions: Counter & Speed ---
    # Removed in v84 per user request.

    def create_reid_config(self):
        """
        ReID Removed per User Request (v67).
        Returns default tracker.
        """
        return "botsort.yaml"

    def init_heatmap(self, colormap=cv2.COLORMAP_JET, imw=640, imh=480):
        """
        Initializes a Heatmap object from ultralytics.solutions.
        """
        try:
            from ultralytics.solutions import Heatmap
            # Heatmap args initialization
            heatmap = Heatmap()
            # Set args usually required if not in init
            # Standard CLI uses set_args, python usage might differ slightly based on version
            # We try to set common args
            heatmap.set_args(colormap=colormap, imw=imw, imh=imh, view_img=False, shape="circle")
            return heatmap
        except ImportError:
            st.warning("Ultralytics Solutions (Heatmap) not available. Update ultralytics.")
            return None
        except Exception as e:
            # st.warning(f"Heatmap init failed: {e}")
            return None

    def process_heatmap(self, heatmap_obj, frame, tracks):
        """
        Updates heatmap with tracks.
        """
        if heatmap_obj is None: return frame
        
        try:
            # Recent API: generate_heatmap(frame, tracks)
            res_frame = heatmap_obj.generate_heatmap(frame, tracks)
            return cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
        except AttributeError:
             return frame
        except Exception as e:
             return frame


import io
import json
import base64
import asyncio
import numpy as np
import cv2
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect

router = APIRouter()
yolo_manager = YOLOManager()

def encode_pil_base64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    if pil_img.mode in ["HSV", "YCbCr"]:
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@router.get("/classes")
async def get_classes(model_name: str = "yolo26n.pt"):
    try:
        actual_model_name = model_name
        if "yolov11" in actual_model_name:
            actual_model_name = actual_model_name.replace("yolov11", "yolo11")
            
        model = yolo_manager.load_model(actual_model_name)
        if model is None: return {}
        return model.names
    except Exception as e:
        return {}

@router.post("/detect")
async def yolo_detect(
    file: UploadFile = File(...),
    model_name: str = Form("yolov8n.pt"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    classes: str = Form("[]"), # JSON string of class IDs
    mode: str = Form("detect"),
    show_boxes: bool = Form(True),
    show_masks: bool = Form(True),
    show_contours: bool = Form(False),
    show_labels: bool = Form(True),
    show_conf: bool = Form(True),
    show_pose: bool = Form(True)
):
    try:
        import json
        selected_classes = json.loads(classes)
        if not selected_classes: selected_classes = None
        else: selected_classes = [int(c) for c in selected_classes]

        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image_pil)
        
        actual_model_name = model_name
        if "yolov11" in actual_model_name:
            actual_model_name = actual_model_name.replace("yolov11", "yolo11")
                
        # Load model
        model = yolo_manager.load_model(actual_model_name)
        if model is None:
            raise HTTPException(status_code=500, detail=f"Failed to load model {actual_model_name}")
            
        # Predict
        results = yolo_manager.predict(model, image_np, conf=conf, iou=iou, classes=selected_classes)
        
        if not results or len(results) == 0:
             return {
                 "image_b64": encode_pil_base64(image_pil),
                 "counts": {},
                 "actual_model_name": actual_model_name
             }
             
        # Plot
        result = results[0]
        plotted_np = yolo_manager.plot_result(
            result, 
            show_conf=show_conf, 
            show_labels=show_labels, 
            show_boxes=show_boxes,
            show_masks=show_masks,
            show_contours=show_contours,
            show_pose=show_pose
        )
        
        # Count classes
        counts = {}
        if hasattr(result, 'boxes'):
            cls_ids = result.boxes.cls.int().cpu().tolist()
            for cls_id in cls_ids:
                name = result.names[cls_id]
                counts[name] = counts.get(name, 0) + 1
        
        return {
            "image_b64": encode_pil_base64(Image.fromarray(cv2.cvtColor(plotted_np, cv2.COLOR_BGR2RGB))),
            "counts": counts,
            "actual_model_name": actual_model_name
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@router.websocket("/ws/detect")
async def websocket_yolo_detect(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    
    try:
        while True:
            # Receive data: { "image": "base64...", "model_name": "...", "conf": 0.25, "classes": [] }
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            img_b64 = payload.get("image")
            if not img_b64: continue
            
            model_name = payload.get("model_name", "yolo26n.pt")
            conf = float(payload.get("conf", 0.25))
            iou = float(payload.get("iou", 0.45))
            selected_classes = payload.get("classes", [])
            if not selected_classes: selected_classes = None
            else: selected_classes = [int(c) for c in selected_classes]
            
            # Tracking options
            mode = payload.get("mode", "detect") # "detect", "segment", "pose", "track"
            show_tracks = payload.get("show_tracks", True)
            
            # Sub-options
            show_boxes = payload.get("show_boxes", True)
            show_masks = payload.get("show_masks", True)
            show_contours = payload.get("show_contours", False)
            show_labels = payload.get("show_labels", True)
            show_conf_text = payload.get("show_conf", True)
            show_pose = payload.get("show_pose", True)
            
            actual_model_name = model_name
            if "yolov11" in actual_model_name:
                actual_model_name = actual_model_name.replace("yolov11", "yolo11")
                
            # Decode b64 image
            header, encoded = img_b64.split(",", 1) if "," in img_b64 else (None, img_b64)
            img_data = base64.b64decode(encoded)
            image_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
            image_np = np.array(image_pil)
            
            # Session history for tracking
            if "track_history" not in websocket.scope:
                websocket.scope["track_history"] = {}
            history = websocket.scope["track_history"]
            
            # Sub-function for executor
            def run_inference():
                model = yolo_manager.load_model(actual_model_name)
                if model is None: 
                    return {"type": "error", "msg": f"模型加载失败: {actual_model_name}"}
                
                # Predict or Track
                if mode == "track":
                    results = yolo_manager.track(model, image_np, conf=conf, iou=iou, classes=selected_classes, persist=True)
                else:
                    results = yolo_manager.predict(model, image_np, conf=conf, iou=iou, classes=selected_classes)
                
                if not results or len(results) == 0:
                    return {"image_b64": img_b64, "counts": {}, "actual_model_name": actual_model_name}
                
                # Plot (unified coloring for tracking)
                result = results[0]
                plotted_np = yolo_manager.plot_result(
                    result, 
                    track_history=history if (mode == "track" and show_tracks) else None,
                    show_masks=show_masks,
                    show_conf=show_conf_text,
                    show_labels=show_labels,
                    show_boxes=show_boxes,
                    show_contours=show_contours,
                    show_pose=show_pose
                )
                
                # Count
                counts = {}
                if hasattr(result, 'boxes'):
                    cls_ids = result.boxes.cls.int().cpu().tolist()
                    for cls_id in cls_ids:
                        name = result.names[cls_id]
                        counts[name] = counts.get(name, 0) + 1
                
                return {
                    "image_b64": encode_pil_base64(Image.fromarray(cv2.cvtColor(plotted_np, cv2.COLOR_BGR2RGB))),
                    "counts": counts,
                    "actual_model_name": actual_model_name
                }

            # Run in executor
            try:
                result_data = await loop.run_in_executor(None, run_inference)
                if result_data:
                    await websocket.send_json(result_data)
            except Exception as e:
                await websocket.send_json({"type": "error", "msg": str(e)})
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"YOLO WS Error: {e}")
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except: pass
