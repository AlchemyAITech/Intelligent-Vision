import cv2
import numpy as np
import os
import streamlit as st
import sys
import torch

# Ensure sam3_repo is in path for imports
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sum3_repo"))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from sam3.model_builder import build_sam3_image_model
    SAM3_AVAILABLE = True
except Exception:
    SAM3_AVAILABLE = False

class SAMManager:
    def __init__(self, model_path="sam3_hvit_b.pt"): # Default to a SAM 3 like name
        self.model_path = model_path
        self.model = None
        self.predictor = None # SAM3InteractiveImagePredictor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "mps" in str(torch.backends.mps.is_available()) and torch.backends.mps.is_available():
            # Use MPS for Mac if available, though SAM 3 might prefer CPU for patched kernels
            self.device = "mps"
        
        # Override to CPU if we are on Mac and want to be safe with our patches
        if sys.platform == "darwin":
            self.device = "cpu"
            
        # Caching
        self.current_image_ref = None # To avoid redundant set_image

    def load_model(self, model_name=None):
        if model_name: 
            self.model_path = model_name
        
        if not SAM3_AVAILABLE:
            st.error("SAM 3 (native) is not available. Please check dependencies and patches.")
            return None

        st.info(f"正在加载原生 SAM 3 模型: {self.model_path} (Device: {self.device})...")
        
        # Check if model exists locally in models/ or current dir
        checkpoint_path = None
        search_paths = [
            os.path.join("models", self.model_path),
            self.model_path,
            os.path.join(repo_path, "checkpoints", self.model_path)
        ]
        for p in search_paths:
            if os.path.exists(p):
                checkpoint_path = p
                break
        
        try:
            # Build SAM 3 with interactivity enabled
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=self.device,
                enable_inst_interactivity=True, # Critical for interactive predictor
                load_from_HF=(checkpoint_path is None) # Download if not found
            )
            
            # The model_builder puts the predictor in model.inst_predictor if enable_inst_interactivity=True
            if hasattr(self.model, "inst_predictor") and self.model.inst_predictor:
                self.predictor = self.model.inst_predictor
            else:
                st.error("模型未包含交互式预测器。请确保 enable_inst_interactivity=True。")
                return None
                
            st.success("SAM 3 加载成功!")
            return self.model
        except Exception as e:
            st.error(f"加载 SAM 3 模型失败: {e}")
            if "Hugging Face" in str(e) or "403" in str(e):
                st.warning("⚠️ 提示: SAM 3 权重可能需要 Hugging Face 访问权限。请确保已在 HF 申请模型权限并使用 `huggingface-cli login` 登录。")
            return None

    def predict_image(self, image, points=None, labels=None, bbox=None, prompt_type=None):
        """
        Generic prediction for Image using SAM 3.
        points: list of [x, y] or [[x,y]]
        labels: list of 1 (foreground) or 0 (background)
        bbox: list of [x1, y1, x2, y2]
        prompt_type: 'everything' for auto-segmentation
        """
        if self.predictor is None: 
            if not self.load_model(): return None
        
        # 1. Set Image (Cached)
        # We compare if it's the same image object or content to avoid redundant encoding
        # In streamlit, image is often a numpy array. 
        # A simple way is to check if it's the same object first.
        if self.current_image_ref is not image:
            self.predictor.set_image(image)
            self.current_image_ref = image
        
        # 2. Prepare Prompts
        point_coords = None
        point_labels = None
        box = None
        
        if points and len(points) > 0:
            point_coords = np.array(points)
            if labels is not None:
                point_labels = np.array(labels)
            else:
                point_labels = np.ones(len(points))
                
        if bbox:
            # SAM 3 predict takes a single box [x1, y1, x2, y2]
            # If multiple boxes, we might need a different approach or loop
            # For now, take the first one if it's a list
            if isinstance(bbox[0], list):
                box = np.array(bbox[0])
            else:
                box = np.array(bbox)
        
        if prompt_type == "everything":
            # Auto-mask generation logic would go here
            # For now, return None or a placeholder if not implemented
            st.warning("SAM 3 自动全图分割功能尚未完全集成。")
            return None
        
        # 3. Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True # Return multiple for ambiguity
        )
        
        # Native SAM returns (masks, scores, logits)
        # masks: (C, H, W), scores: (C,), logits: (C, 256, 256)
        
        # To maintain compatibility with convolution_app's `results[0].masks.data`,
        # we wrap it in a mock object structure.
        class MockMasks:
            def __init__(self, data):
                self.data = torch.from_numpy(data)
        
        class MockResult:
            def __init__(self, masks_data):
                self.masks = MockMasks(masks_data)
        
        # convolution_app expects results[0].masks.data
        return [MockResult(masks)]

    def overlay_mask(self, image, mask, color=(0, 255, 0), alpha=0.5):
        """
        Composites a binary mask onto an image.
        image: HxWx3 (RGB)
        mask: HxW (Binary 0/1 or False/True)
        color: RGB tuple
        """
        if mask is None: return image
        
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
        mask = mask.astype(bool)
        
        overlay = image.copy()
        # Ensure color is numpy array for broadcasting
        overlay[mask] = np.array(color, dtype=np.uint8)
        
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def track_video(self, video_path, points=None, labels=None, bbox=None):
        """
        Video tracking placeholder for SAM 3.
        """
        st.warning("SAM 3 视频追踪功能尚未在 SAMManager 中完全集成。")
        return None

    def auto_segment(self, image):
        """
        Placeholder for Automatic Mask Generation.
        """
        st.warning("SAM 3 自动分割功能尚未在 SAMManager 中完全集成。")
        return None
