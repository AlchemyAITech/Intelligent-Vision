import io
import base64
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy import signal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional

# ==========================================
# 核心处理函数 (卷积算法集成)
# ==========================================

def apply_kernel_2d(image_2d, kernel):
    return signal.convolve2d(image_2d, kernel, mode='same', boundary='symm')

def convolve_rgb(image_pil, kernel):
    """
    对 RGB 三通道分别卷积
    """
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
        
    r, g, b = image_pil.split()
    r_out = apply_kernel_2d(np.array(r), kernel)
    g_out = apply_kernel_2d(np.array(g), kernel)
    b_out = apply_kernel_2d(np.array(b), kernel)
    
    r_new = Image.fromarray(np.clip(r_out, 0, 255).astype(np.uint8))
    g_new = Image.fromarray(np.clip(g_out, 0, 255).astype(np.uint8))
    b_new = Image.fromarray(np.clip(b_out, 0, 255).astype(np.uint8))
    return Image.merge('RGB', (r_new, g_new, b_new))

# --- 卷积核定义 ---
KERNELS = {
    'Blur (模糊)': {
        'Box Blur 3x3': np.ones((3, 3)) / 9.0,
        'Box Blur 5x5': np.ones((5, 5)) / 25.0,
        'Box Blur 15x15': np.ones((15, 15)) / (15.0*15.0),
        'Gaussian Blur 3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0,
        'Gaussian Blur 5x5': np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0
    },
    'Sharpen (锐化)': {
        'Standard 3x3': np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]]),
        'Strong 3x3': np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]]),
        'Unsharp Masking': np.array([[-1, -2, -1], [-2, 28, -2], [-1, -2, -1]]) / 16.0,
    },
    'Lightness (亮度)': {
        'Standard 3x3': np.array([[ 0, 0,  0], [0,  1.2, 0], [ 0, 0,  0]]),
        'Strong 3x3': np.array([[ 0, 0,  0], [0,  1.5, 0], [ 0, 0,  0]]),
        'Unsharp Masking': np.array([[ 0, 0,  0], [0,  2, 0], [ 0, 0,  0]]),
    },
    'Edge Detection (边缘)': {
        'Sobel X (Vertical)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Sobel Y (Horizontal)': np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]),
        'Prewitt X': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'Prewitt Y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'Laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        'Laplacian 5x5': np.array([[0, 1, 2, 1, 0], [1, 2, 4, 2, 1], [2, 4, -20, 4, 2], [1, 2, 4, 2, 1], [0, 1, 2, 1, 0]]),
        'Outline': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Outline 5x5': np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])
    },
    'Artistic (艺术)': {
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        'Oil Texture': np.array([[1, 2, 1], [2, 8, 2],[1, 2, 1]]) / 20.0,
        'Light from Top Left': np.array([[ 1,  1,  0],[ 1,  0, -1],[ 0, -1, -1]]),
        'Cartoon Edge': np.array([[ 1,  1,  1],[ 1, -8,  1],[ 1,  1,  1]]),
        'Identity (无操作)': np.zeros((3,3))
    }
}
KERNELS['Artistic (艺术)']['Identity (无操作)'][1,1] = 1

router = APIRouter()

def encode_pil_base64(pil_img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    # convert modes that cannot be saved directly
    if pil_img.mode in ["HSV", "YCbCr"]:
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@router.get("/kernels")
async def get_kernels():
    """Returns the available kernels grouped by category."""
    # Convert numpy arrays to lists for JSON serialization
    serialized_kernels = {}
    for cat, kernels in KERNELS.items():
        serialized_kernels[cat] = {
            name: kernel.tolist() for name, kernel in kernels.items()
        }
    return serialized_kernels

@router.post("/color_space")
async def process_color_space(
    file: UploadFile = File(...),
    target_space: str = Form("RGB (Original)") # "Grayscale (L)", "YCbCr", "HSV", "RGB (Original)"
):
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if target_space == "Grayscale (L)":
            converted_img = image_pil.convert('L')
            channels = [converted_img]
            channel_names = ['Gray']
        elif target_space == "YCbCr":
            converted_img = image_pil.convert('YCbCr')
            channels = converted_img.split()
            channel_names = ['Y (Luma)', 'Cb', 'Cr']
        elif target_space == "HSV":
            converted_img = image_pil.convert('HSV')
            channels = converted_img.split()
            channel_names = ['Hue', 'Saturation', 'Value']
        else:
            converted_img = image_pil
            channels = converted_img.split()
            channel_names = ['Red', 'Green', 'Blue']
            
        # Get center 20x20 matrices
        img_arr = np.array(converted_img)
        h, w = img_arr.shape[:2]
        size = 20
        cy, cx = h // 2, w // 2
        half = size // 2
        
        r_start = max(0, cy - half)
        r_end = r_start + size
        c_start = max(0, cx - half)
        c_end = c_start + size
        
        crop_arr = img_arr[r_start:r_end, c_start:c_end]
        matrices = []
        if len(crop_arr.shape) == 3:
            for i in range(crop_arr.shape[2]):
                matrices.append(crop_arr[:, :, i].tolist())
        else:
            matrices.append(crop_arr.tolist())
            
        # Draw red box on the output image
        out_img = converted_img.convert('RGB')
        draw = ImageDraw.Draw(out_img)
        draw.rectangle([c_start, r_start, c_end, r_end], outline="red", width=3)
            
        return {
            "image_b64": encode_pil_base64(out_img),
            "channel_names": channel_names,
            "matrices": matrices
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/convolution")
async def process_convolution(
    file: UploadFile = File(...),
    process_mode: str = Form("RGB (彩色)"), # "RGB (彩色)" or "Grayscale (灰度)"
    category: str = Form(...),
    kernel_name: str = Form(...),
    invert_color: str = Form("false")
):
    try:
        invert_color = invert_color.lower() == "true"
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if category not in KERNELS or kernel_name not in KERNELS[category]:
            raise ValueError("Invalid kernel selected.")
            
        kernel = KERNELS[category][kernel_name]
        is_gray_input = (process_mode == "Grayscale (灰度)")
        
        if is_gray_input:
            input_img = image_pil.convert('L')
            res_arr = apply_kernel_2d(np.array(input_img), kernel)
            res_arr = np.clip(res_arr, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(res_arr)
            if invert_color:
                result_img = ImageOps.invert(result_img)
        else:
            input_img = image_pil
            result_img = convolve_rgb(input_img, kernel)
            if invert_color:
                if result_img.mode == 'RGBA': result_img = result_img.convert('RGB')
                result_img = ImageOps.invert(result_img)
                
        return {
            "image_b64": encode_pil_base64(result_img)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
