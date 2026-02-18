import numpy as np
from PIL import Image, ImageOps
from scipy import signal

# ==========================================
# 核心处理函数 (图像处理)
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
