from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import shutil
import yaml

router = APIRouter()

class ClassificationTagRequest(BaseModel):
    project_name: str
    images: List[str]
    tag: str

class DetectionLabelRequest(BaseModel):
    project_name: str
    image: str
    boxes: List[dict] # {class_id: 0, x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.2}

@router.post("/classify/apply_tags")
async def apply_classification_tags(req: ClassificationTagRequest):
    """
    将图像划入 Ultralytics 分类格式的目录层级中。
    Classification: dataset/train/{class_name}/img.jpg
    """
    base_dir = os.path.join("data", "projects", req.project_name, "dataset", "train")
    target_dir = os.path.join(base_dir, req.tag)
    os.makedirs(target_dir, exist_ok=True)
    
    # 模拟数据挂载 (在真实全量链路中，这里会执行 shutil.copy 从暂存区入库)
    for img_name in req.images:
        target_file = os.path.join(target_dir, img_name)
        # 用空文件模拟入站
        with open(target_file, "w") as f:
            f.write(f"# MOCK DATASET BIND: {img_name}")

    return {
        "status": "success", 
        "message": f"成功将 {len(req.images)} 张样本打上类别标签: {req.tag}", 
        "target_dir": target_dir
    }

@router.post("/detection/apply_labels")
async def apply_detection_labels(req: DetectionLabelRequest):
    """
    接收目标检测的坐标并转录为 Ultralytics 标准 YOLO 格式 (xxx.txt)。
    Detection:
        dataset/images/train/img.jpg
        dataset/labels/train/img.txt -> 0 0.5 0.5 0.2 0.2
    """
    proj_dir = os.path.join("data", "projects", req.project_name, "dataset")
    img_dir = os.path.join(proj_dir, "images", "train")
    lbl_dir = os.path.join(proj_dir, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 写入 YOLO .txt 格式
    base_name = os.path.splitext(req.image)[0]
    txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
    
    with open(txt_path, "w") as f:
        for box in req.boxes:
            # YOLO format: class x_center y_center width height (normalized)
            f.write(f"{box['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}\n")

    return {
        "status": "success",
        "message": f"成功录入 1 个检测样本，包含 {len(req.boxes)} 个目标框。",
        "label_path": txt_path
    }

@router.post("/init_yaml")
async def init_dataset_yaml(project_name: str, classes: List[str]):
    """
    自动生成传递给 Ultralytics 训练器的 dataset.yaml 配置图谱。
    """
    proj_dir = os.path.join("data", "projects", project_name)
    os.makedirs(proj_dir, exist_ok=True)
    yaml_path = os.path.join(proj_dir, "dataset.yaml")

    yaml_data = {
        "path": os.path.abspath(os.path.join(proj_dir, "dataset")),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)}
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    return {
        "status": "success",
        "message": "生成 dataset.yaml 架构地图完成",
        "yaml_path": yaml_path
    }
