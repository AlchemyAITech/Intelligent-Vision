from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import shutil
import yaml

router = APIRouter()

class CreateDatasetRequest(BaseModel):
    project_name: str
    description: str = ""

class ClassificationTagRequest(BaseModel):
    project_name: str
    images: List[str]
    tag: str

import json

@router.get("/list")
async def list_datasets():
    """获取所有已创建的数据仓库（项目库）"""
    projects_dir = os.path.join("data", "projects")
    os.makedirs(projects_dir, exist_ok=True)
    
    datasets = []
    for p in os.listdir(projects_dir):
        p_path = os.path.join(projects_dir, p)
        if os.path.isdir(p_path):
            meta_path = os.path.join(p_path, "meta.json")
            description = ""
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        description = meta.get("description", "")
                except:
                    pass
            
            # Simple count of images
            img_count = 0
            img_dir = os.path.join(p_path, "dataset", "images", "train")
            if os.path.exists(img_dir):
                img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                
            datasets.append({
                "name": p,
                "description": description,
                "image_count": img_count
            })
            
    return {"status": "success", "data": datasets}

@router.post("/create")
async def create_dataset(req: CreateDatasetRequest):
    """创建新的数据仓库（项目库）"""
    p_path = os.path.join("data", "projects", req.project_name)
    if os.path.exists(p_path):
        raise HTTPException(status_code=400, detail="同名数据仓库已存在")
        
    os.makedirs(p_path, exist_ok=True)
    
    # 建立深度学所需的基础结构
    os.makedirs(os.path.join(p_path, "dataset", "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(p_path, "dataset", "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(p_path, "dataset", "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(p_path, "dataset", "labels", "val"), exist_ok=True)
    
    # 保存元数据
    meta_path = os.path.join(p_path, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"description": req.description}, f, ensure_ascii=False)
        
    return {"status": "success", "message": f"成功创立数据仓: {req.project_name}"}

@router.delete("/{project_name}")
async def delete_dataset(project_name: str):
    """危险：彻底抹除某个数据仓库"""
    p_path = os.path.join("data", "projects", project_name)
    if not os.path.exists(p_path):
        raise HTTPException(status_code=404, detail="仓库不存在")
        
    try:
        shutil.rmtree(p_path)
        return {"status": "success", "message": f"已销毁数据仓: {project_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


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
