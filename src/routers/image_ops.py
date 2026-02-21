import io
import base64
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional

# Import local modules
from src.image_ops import apply_kernel_2d, convolve_rgb, KERNELS

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
