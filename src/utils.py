import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import os

def draw_red_box(image_pil, size=20):
    """
    在图像中心画一个红色的矩形框
    """
    if image_pil.mode != 'RGB':
        img_draw = image_pil.convert('RGB')
    else:
        img_draw = image_pil.copy()
        
    draw = ImageDraw.Draw(img_draw)
    w, h = img_draw.size
    cx, cy = w // 2, h // 2
    half = size // 2
    
    xy = [cx - half, cy - half, cx + half - 1, cy + half - 1]
    draw.rectangle(xy, outline="red", width=2)
    return img_draw

# 获取图像中心区域的矩阵数据
def get_center_matrix(image_pil, size=20):
    img_arr = np.array(image_pil)
    h, w = img_arr.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    
    r_start = max(0, cy - half)
    r_end = r_start + size
    c_start = max(0, cx - half)
    c_end = c_start + size
    
    crop = img_arr[r_start:r_end, c_start:c_end]
    
    if len(crop.shape) == 3:
        channels_data = []
        for i in range(crop.shape[2]):
            channels_data.append(pd.DataFrame(crop[:, :, i]))
        return channels_data
    else:
        return [pd.DataFrame(crop)]

def load_source_image(key):
    """
    UI Component to load image from Upload, Local, or Webcam.
    Matches the YOLO tab style: Vertical layout.
    """
    source_type = st.radio("选择输入源", ["🖼️ 图片上传 (Upload)", "📂 本地文件 (Local)", "📷 摄像头 (Webcam)"], 
                           key=f"{key}_type")
    
    image = None
    
    if source_type == "🖼️ 图片上传 (Upload)":
        uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg", "webp", "bmp"], key=f"{key}_upload")
        if uploaded_file:
            try:
                # Save to local
                upload_dir = "images/uploads"
                if not os.path.exists(upload_dir): os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                image = Image.open(file_path).convert('RGB')
            except Exception as e:
                st.error(f"无法加载上传的图片: {e}")
                
    elif source_type == "📂 本地文件 (Local)":
        image_dir = "images"
        if not os.path.exists(image_dir): os.makedirs(image_dir, exist_ok=True)
        
        try:
            files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
            if files:
                selected_file = st.selectbox("选择文件", files, key=f"{key}_local")
                if selected_file:
                    image = Image.open(os.path.join(image_dir, selected_file)).convert('RGB')
            else:
                st.info(f"请将图片放入 `{image_dir}` 文件夹")
        except Exception as e:
            st.error(f"读取本地文件出错: {e}")
            
    elif source_type == "📷 摄像头 (Webcam)":
        cam_file = st.camera_input("拍照", key=f"{key}_cam")
        if cam_file:
            try:
                image = Image.open(cam_file).convert('RGB')
            except Exception as e:
                st.error(f"摄像头图片读取失败: {e}")

    # Fallback / Default Image Logic
    if image is None:
         # Try to load demo.jpg as default if nothing is selected/uploaded yet?
         # Or just return None and let the main app handle "Please Select".
         # Existing logic had a fallback generator. Let's keep it but only if explicitly needed?
         # User might want a clean state. But "Convolution Lab" usually starts with an image.
         # Let's try to load 'demo.jpg' if exists and no interaction happened?
         # It's hard to know if interaction happened. 
         # Let's just return None to let user pick, BUT helper functions in app might expect an image.
         # The original code returned a generated image if local dir was empty or nothing selected.
         # Let's keep the demo.jpg fallback for smoother UX.
         
         demo_path = os.path.join("images", "demo.jpg")
         if os.path.exists(demo_path):
             # Only load default if we are in Local mode? 
             # Or always? If I just switched to Upload and havn't uploaded, showing text is better.
             # Let's return None and let the UI handle "Please upload".
             pass
         else:
             pass
             
    return image
