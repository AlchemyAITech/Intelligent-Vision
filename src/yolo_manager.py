import os
import cv2
import sys
import streamlit as st
from ultralytics import YOLO, SAM
import numpy as np

class YOLOManager:
    """
    Manages YOLO model loading, inference, and result visualization.
    """
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path, model_name):
    """
    Helper function to load model with caching.
    Ensures model file is in the correct directory.
    """
    try:
        # 1. If exists at target, load
        if os.path.exists(model_path):
             return SAM(model_path) if "sam" in model_name.lower() else YOLO(model_path)
             
        # 2. Check if exists in root (cwd)
        root_path = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        if os.path.exists(root_path):
             import shutil
             shutil.move(root_path, model_path)
             return SAM(model_path) if "sam" in model_name.lower() else YOLO(model_path)

        # 3. Not found, attempt download
        # We use a temp instance to trigger download to CWD
        try:
            # This triggers download to CWD
            _ = SAM(model_name) if "sam" in model_name.lower() else YOLO(model_name)
        except Exception:
            pass
            
        # 4. Check root again after download attempt and move
        if os.path.exists(root_path):
             import shutil
             shutil.move(root_path, model_path)
             return SAM(model_path) if "sam" in model_name.lower() else YOLO(model_path)
             
        # 5. Last resort: just try loading (maybe it downloaded somewhere else or failed move)
        return SAM(model_name) if "sam" in model_name.lower() else YOLO(model_name)

    except Exception as e:
        # st.error(f"Error loading model {model_name}: {e}")
        return None

class YOLOManager:
    """
    Manages YOLO model loading, inference, and result visualization.
    """
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self, model_name):
        """
        Loads the model using the cached helper.
        """
        model_path = os.path.join(self.model_dir, model_name)
        model = load_model_cached(model_path, model_name)
        if model is None:
             st.error(f"Failed to load model: {model_name}")
        return model

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
            # YOLO inference with retina_masks for pixel-perfect alignment
            results = model.predict(source, conf=conf, iou=iou, classes=classes, verbose=False, retina_masks=True)
            
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
            # YOLO Tracking with retina_masks for pixel-perfect alignment
            results = model.track(source, persist=persist, conf=conf, iou=iou, classes=classes, verbose=False, tracker=tracker, retina_masks=True)
            
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

    def plot_result(self, result, show_conf=True, show_labels=True, show_boxes=True, show_masks=True, show_contours=False, track_history=None):
        """
        Plots the result and returns an RGB image.
        Unified ID-based coloring for Tracking (Box + Mask + Trail).
        Class-based coloring for Static Detection.
        """
        try:
            import cv2
            import numpy as np
            from ultralytics.utils.plotting import colors

            # Determine Mode
            # If tracking (history + IDs present), we take Full Control for ID-based coloring.
            is_tracking = track_history is not None and hasattr(result, 'boxes') and result.boxes.id is not None
            
            # --- 1. Static Mode (Class-based) ---
            if not is_tracking:
                 # Standard Ultralytics Plotting
                 res_plotted = result.plot(conf=show_conf, labels=show_labels, boxes=show_boxes, masks=show_masks)
                 
                 # Optional: Add Contours on top if requested (Class-based)
                 if show_contours and result.masks is not None:
                      masks_data = result.masks.data.cpu().numpy()
                      cls_ids = result.boxes.cls.int().cpu().tolist()
                      for i, mask in enumerate(masks_data):
                           cls_id = int(cls_ids[i]) if i < len(cls_ids) else 0
                           color = colors(cls_id, bgr=True)
                           
                           m_bin = (mask > 0.5).astype(np.uint8) * 255
                           cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                           for cnt in cnts:
                                if cv2.contourArea(cnt) < 10: continue
                                epsilon = 0.0002 * cv2.arcLength(cnt, True)
                                approx = cv2.approxPolyDP(cnt, epsilon, True)
                                cv2.polylines(res_plotted, [approx], True, color, 2, lineType=cv2.LINE_AA)
                 
                 return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB) if res_plotted is not None else None

            # --- 2. Tracking Mode (ID-based, Unified) ---
            # We assume full control. Draw on copy of original image (or clean plot)
            # result.plot(boxes=False, ...) returns essentially the original image with some preprocessing?
            # Safer to use result.orig_img if available, or just use plot(all=False)
            res_plotted = result.plot(conf=False, labels=False, boxes=False, masks=False, probs=False)
            
            # Data Preparation
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.int().cpu().tolist()
            conf_scores = result.boxes.conf.float().cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            
            masks_data = None
            if hasattr(result, 'masks') and result.masks is not None:
                 masks_data = result.masks.data.cpu().numpy() # Full res masks (retina_masks=True in track params)

            # Iterate Objects
            for i, track_id in enumerate(track_ids):
                 # Generate ID Color (Consistent)
                 np.random.seed(int(track_id))
                 id_color = np.random.randint(0, 255, size=3).tolist() # BGR
                 
                 # 1. Update & Draw Trail (History)
                 center = (float((boxes_xyxy[i][0] + boxes_xyxy[i][2]) / 2), float((boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2))
                 track = track_history.get(track_id, [])
                 track.append(center)
                 if len(track) > 30: track.pop(0) # Limit history
                 track_history[track_id] = track
                 
                 points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                 if len(points) > 1:
                      for j in range(len(points) - 1):
                           thickness = int(np.sqrt(float(j + 1)) * 1.5)
                           cv2.line(res_plotted, tuple(points[j][0]), tuple(points[j+1][0]), id_color, thickness)
                 
                 # 2. Draw Mask (If enabled)
                 if show_masks and masks_data is not None and i < len(masks_data):
                      m_bin = (masks_data[i] > 0.5).astype(np.uint8) * 255
                      
                      # Alpha Blend Mask
                      # Create colored mask
                      colored_mask = np.zeros_like(res_plotted, dtype=np.uint8)
                      # Fill mask region with ID color
                      # We can use m_bin as mask
                      colored_mask[m_bin == 255] = id_color
                      
                      # Blend: dst = src1*alpha + src2*beta + gamma
                      # We only blend where mask is present to preserve background speed? 
                      # Or just global blend on ROI? Global blend entire frame is slow.
                      # Optimization: ROI blend or cv2.addWeighted on full frame. 
                      # For robustness, we'll do: where mask, pixel = pixel*0.6 + color*0.4
                      
                      # Using mask indices is faster
                      mask_bool = (m_bin == 255)
                      res_plotted[mask_bool] = (res_plotted[mask_bool] * 0.6 + np.array(id_color) * 0.4).astype(np.uint8)

                 # 3. Draw Contour (If enabled or if Mask is enabled we usually show border)
                 # User asked "Mask color same as box", so we use ID color for border too.
                 if (show_masks or show_contours) and masks_data is not None and i < len(masks_data):
                      m_bin = (masks_data[i] > 0.5).astype(np.uint8) * 255
                      cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                      for cnt in cnts:
                           if cv2.contourArea(cnt) < 10: continue
                           epsilon = 0.0002 * cv2.arcLength(cnt, True)
                           approx = cv2.approxPolyDP(cnt, epsilon, True)
                           cv2.polylines(res_plotted, [approx], True, id_color, 2, lineType=cv2.LINE_AA)

                 # 4. Draw Box & Label
                 if show_boxes:
                      x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                      cv2.rectangle(res_plotted, (x1, y1), (x2, y2), id_color, 2)
                      
                      if show_labels:
                           cls_idx = cls_ids[i]
                           label = result.names[cls_idx]
                           text = f"{label} {track_id}" 
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
