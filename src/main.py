from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uvicorn

# Import Routers
from src.routers import sam, image_ops, cnn, common, yolo, face

app = FastAPI(title="Intelligent Vision - AI Labs")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(sam.router, prefix="/api/sam", tags=["SAM"])
app.include_router(image_ops.router, prefix="/api/image", tags=["ImageOps"])
app.include_router(cnn.router, prefix="/api/cnn", tags=["CNN"])
app.include_router(common.router, prefix="/api/common", tags=["Common"])
app.include_router(yolo.router, prefix="/api/yolo", tags=["YOLO"])
app.include_router(face.router, prefix="/api/face", tags=["Face"])

# Serve Static Files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
images_path = os.path.join(os.path.dirname(__file__), "..", "images")
bank_path = os.path.join(os.path.dirname(__file__), "..", "data", "face_bank")

# Mount specialized paths first
if os.path.exists(images_path):
    app.mount("/images", StaticFiles(directory=images_path), name="images")

video_path_dir = os.path.join(os.path.dirname(__file__), "..", "video")
if os.path.exists(video_path_dir):
    app.mount("/video", StaticFiles(directory=video_path_dir), name="video")

if os.path.exists(bank_path):
    app.mount("/face_bank", StaticFiles(directory=bank_path), name="face_bank")

uploads_path = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(uploads_path, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_path), name="uploads")

# Mount frontend at root (html, js, css)
# html(index.html) is served by root route below, but other assets need to be 
# accessible at /...
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
