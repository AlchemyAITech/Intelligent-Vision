from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uvicorn

# Import Routers
from src.routers import image_ops, cnn, common, yolo, face, dataset, training, analytica, model_project

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
app.include_router(image_ops.router, prefix="/api/image", tags=["ImageOps"])
app.include_router(cnn.router, prefix="/api/cnn", tags=["CNN"])
app.include_router(common.router, prefix="/api/common", tags=["Common"])
app.include_router(yolo.router, prefix="/api/yolo", tags=["YOLO"])
app.include_router(face.router, prefix="/api/face", tags=["Face"])
app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(analytica.router, prefix="/api/analytica", tags=["Analytica"])
app.include_router(model_project.router, prefix="/api/model_project", tags=["Model Project"])

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

# Explicitly serve index.html with no-cache to ensure frontend updates propagate
@app.get("/")
@app.get("/index.html")
async def serve_index():
    return FileResponse(
        os.path.join(frontend_path, "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=32100, reload=True)
