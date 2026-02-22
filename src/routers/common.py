from fastapi import APIRouter
import os
import glob

router = APIRouter()

@router.get("/local_images")
async def get_local_images():
    """Return a list of local images from the 'images' directory."""
    image_dir = "images"
    if not os.path.exists(image_dir):
        return {"files": []}
    
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
    return {"files": files}

@router.get("/local_videos")
async def get_local_videos():
    """Return a list of local videos from the 'video' directory."""
    video_dir = "video"
    if not os.path.exists(video_dir):
        return {"files": []}
    
    files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))]
    return {"files": files}
