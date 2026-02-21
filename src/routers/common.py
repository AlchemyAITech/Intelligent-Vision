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
