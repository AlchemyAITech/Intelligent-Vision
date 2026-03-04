from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import yaml
import json

router = APIRouter()

class CreateDatasetRequest(BaseModel):
    project_name: str
    description: str = ""

class RenameProjectRequest(BaseModel):
    new_name: str

class ClassificationTagRequest(BaseModel):
    project_name: str
    images: List[str]
    tag: str

class CategoryRequest(BaseModel):
    name: str

class RenameCategoryRequest(BaseModel):
    new_name: str

class VersionRequest(BaseModel):
    version_name: str

class BatchSplitRequest(BaseModel):
    images: List[str]
    split: str

class BatchTagRequest(BaseModel):
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
                img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
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

@router.post("/rename/{project_name}")
async def rename_dataset(project_name: str, req: RenameProjectRequest):
    """修改数据仓库（项目库）名称"""
    p_path = os.path.join("data", "projects", project_name)
    new_p_path = os.path.join("data", "projects", req.new_name)
    
    if not os.path.exists(p_path):
        raise HTTPException(status_code=404, detail="仓库不存在")
    if os.path.exists(new_p_path):
        raise HTTPException(status_code=400, detail="同名数据仓库已存在")
        
    try:
        os.rename(p_path, new_p_path)
        return {"status": "success", "message": f"成功重命名为: {req.new_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重命名失败: {str(e)}")


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

# ---------------------------------------------------------
# Advanced Classification & Data Management Features
# ---------------------------------------------------------

def get_meta(project_name: str):
    meta_path = os.path.join("data", "projects", project_name, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"description": "", "categories": []}

def save_meta(project_name: str, meta: dict):
    meta_path = os.path.join("data", "projects", project_name, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

@router.get("/{project_name}/categories")
async def get_categories(project_name: str):
    meta = get_meta(project_name)
    return {"status": "success", "data": meta.get("categories", [])}

@router.post("/{project_name}/categories")
async def add_category(project_name: str, req: CategoryRequest):
    meta = get_meta(project_name)
    cats = meta.get("categories", [])
    if req.name not in cats:
        cats.append(req.name)
        meta["categories"] = cats
        save_meta(project_name, meta)
    return {"status": "success", "data": cats}

@router.put("/{project_name}/categories/{old_name}")
async def rename_category(project_name: str, old_name: str, req: RenameCategoryRequest):
    meta = get_meta(project_name)
    cats = meta.get("categories", [])
    if old_name in cats:
        idx = cats.index(old_name)
        cats[idx] = req.new_name
        meta["categories"] = cats
        save_meta(project_name, meta)
    return {"status": "success", "data": cats}

@router.delete("/{project_name}/categories/{name}")
async def delete_category(project_name: str, name: str):
    meta = get_meta(project_name)
    cats = meta.get("categories", [])
    if name in cats:
        cats.remove(name)
        meta["categories"] = cats
        save_meta(project_name, meta)
    return {"status": "success", "data": cats}

DATASETS_DIR = os.path.join("data", "projects")

class VersionCreateRequest(BaseModel):
    version: str
    description: Optional[str] = ""

@router.get("/{project_name}/versions")
async def get_dataset_versions(project_name: str):
    project_dir = os.path.join(DATASETS_DIR, project_name)
    annotations_dir = os.path.join(project_dir, "annotations")
    
    if not os.path.exists(annotations_dir):
        return {"versions": []}
        
    versions = []
    for f in os.listdir(annotations_dir):
        if f.endswith(".json") and f != "meta.json":
            version_name = f.replace(".json", "")
            
            # Extract description from file if present (simple implementation)
            description = ""
            try:
                filepath = os.path.join(annotations_dir, f)
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    description = data.get("_description", "")
            except Exception:
                pass
                
            versions.append({
                "version": version_name,
                "description": description
            })
    return {"status": "success", "data": versions}

@router.post("/{project_name}/versions")
async def create_dataset_version(project_name: str, req: VersionCreateRequest):
    project_dir = os.path.join(DATASETS_DIR, project_name)
    annotations_dir = os.path.join(project_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True) # Ensure annotations directory exists
    
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    version_file = os.path.join(annotations_dir, f"{req.version}.json")
    if os.path.exists(version_file):
        raise HTTPException(status_code=400, detail="Version already exists")
        
    # Initialize empty version file with metadata
    with open(version_file, "w", encoding="utf-8") as f:
        json.dump({
            "_description": req.description,
            "images": {}
        }, f, indent=4, ensure_ascii=False)
        
    return {"status": "success", "version": req.version, "description": req.description}

def get_splits_file(project_name: str):
    s_path = os.path.join("data", "projects", project_name, "splits.json")
    if not os.path.exists(s_path):
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    return s_path

def get_annotations_dir(project_name: str):
    path = os.path.join("data", "projects", project_name, "annotations")
    os.makedirs(path, exist_ok=True)
    return path

@router.get("/{project_name}/stats")
async def get_dataset_stats(project_name: str, version: str = "v1"):
    proj_dir = os.path.join("data", "projects", project_name, "dataset")
    
    total = 0
    if os.path.exists(proj_dir):
        for root, _, files in os.walk(proj_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    total += 1
                    
    s_path = get_splits_file(project_name)
    with open(s_path, "r", encoding="utf-8") as f:
        splits_map = json.load(f)
        
    ann_dir = get_annotations_dir(project_name)
    v_path = os.path.join(ann_dir, f"{version}.json")
    ann_map = {}
    if os.path.exists(v_path):
        with open(v_path, "r", encoding="utf-8") as f:
            ann_map = json.load(f)
            
    stats = {
        "total": total,
        "splits": {
            "train": 0, "val": 0, "test": 0, "unassigned": 0
        },
        "annotated": 0,
        "unannotated": 0
    }
    
    for img, sp in splits_map.items():
        if sp in stats["splits"]:
            stats["splits"][sp] += 1
            
    stats["splits"]["unassigned"] = total - sum(stats["splits"].values())
    stats["annotated"] = len(ann_map)
    stats["unannotated"] = total - stats["annotated"]
    
    return {"status": "success", "data": stats}

@router.get("/{project_name}/images")
async def get_dataset_images(project_name: str, version: str = "v1", split: str = "all", annotated: str = "all", category: str = "all", search: str = "", page: int = 1, page_size: int = 50):
    proj_dir = os.path.join("data", "projects", project_name, "dataset")
    
    all_images = []
    # If project is empty, dataset might not exist yet, we handled os.makedirs on create
    if os.path.exists(proj_dir):
        for root, _, files in os.walk(proj_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(file)
                
    all_images.sort()
    
    s_path = get_splits_file(project_name)
    with open(s_path, "r", encoding="utf-8") as f:
        splits_map = json.load(f)
        
    ann_dir = get_annotations_dir(project_name)
    v_path = os.path.join(ann_dir, f"{version}.json")
    ann_map = {}
    if os.path.exists(v_path):
        with open(v_path, "r", encoding="utf-8") as f:
            ann_map = json.load(f)
            
    filtered = []
    for img in all_images:
        img_split = splits_map.get(img, "unassigned")
        img_ann = ann_map.get(img, None)
        
        if split != "all":
            if split == "unassigned" and img_split != "unassigned": continue
            elif split != "unassigned" and img_split != split: continue
            
        if annotated == "annotated" and not img_ann: continue
        if annotated == "unannotated" and img_ann: continue
        
        if category != "all":
            if not img_ann or img_ann != category:
                continue
        
        if search:
            s_lower = search.lower()
            if s_lower not in img.lower() and (not img_ann or s_lower not in img_ann.lower()):
                continue
                
        filtered.append({
            "name": img,
            "split": img_split,
            "category": img_ann
        })
        
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]
    
    return {
        "status": "success",
        "data": paginated,
        "total": total,
        "page": page,
        "page_size": page_size
    }

@router.post("/{project_name}/images/split")
async def batch_split(project_name: str, req: BatchSplitRequest):
    s_path = get_splits_file(project_name)
    with open(s_path, "r", encoding="utf-8") as f:
        splits_map = json.load(f)
        
    for img in req.images:
        splits_map[img] = req.split
        
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(splits_map, f, ensure_ascii=False, indent=2)
        
    return {"status": "success", "message": f"Assigned {len(req.images)} images to {req.split}"}

@router.post("/{project_name}/annotations/{version}/batch")
async def batch_tag(project_name: str, version: str, req: BatchTagRequest):
    ann_dir = get_annotations_dir(project_name)
    v_path = os.path.join(ann_dir, f"{version}.json")
    ann_map = {}
    if os.path.exists(v_path):
        with open(v_path, "r", encoding="utf-8") as f:
            ann_map = json.load(f)
            
    for img in req.images:
        ann_map[img] = req.tag
        
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(ann_map, f, ensure_ascii=False, indent=2)
        
        
    return {"status": "success", "message": f"Tagged {len(req.images)} images as {req.tag}"}

def find_image_path(project_name: str, image_name: str):
    ds_dir = os.path.join("data", "projects", project_name, "dataset")
    # Quick check in known directories
    known_dirs = [
        os.path.join(ds_dir, "images", "train"),
        os.path.join(ds_dir, "images", "val"),
        os.path.join(ds_dir, "images", "test"),
        os.path.join(ds_dir, "images", "unassigned"),
        os.path.join(ds_dir, "images"),
        ds_dir
    ]
    for d in known_dirs:
        p = os.path.join(d, image_name)
        if os.path.exists(p):
            return p
            
    # Fallback to os.walk
    for root, _, files in os.walk(ds_dir):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

@router.get("/{project_name}/image/{image_name}")
async def get_image(project_name: str, image_name: str):
    image_path = find_image_path(project_name, image_name)
    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@router.post("/{project_name}/images")
async def upload_images(project_name: str, files: List[UploadFile] = File(...)):
    ds_dir = os.path.join("data", "projects", project_name, "dataset")
    unassigned_dir = os.path.join(ds_dir, "images", "unassigned")
    os.makedirs(unassigned_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(unassigned_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            
    return {"status": "success", "message": f"Uploaded {len(saved_files)} images", "files": saved_files}

class DeleteImagesRequest(BaseModel):
    images: List[str]

@router.delete("/{project_name}/images")
async def delete_images(project_name: str, req: DeleteImagesRequest):
    # Remove files physically
    deleted_count = 0
    for img in req.images:
        path = find_image_path(project_name, img)
        if path and os.path.exists(path):
            os.remove(path)
            deleted_count += 1
            
    # Cleanup splits
    s_path = get_splits_file(project_name)
    if os.path.exists(s_path):
        with open(s_path, "r", encoding="utf-8") as f:
            splits_map = json.load(f)
        for img in req.images:
            splits_map.pop(img, None)
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(splits_map, f, ensure_ascii=False, indent=2)
            
    # Note: We don't strictly need to cleanup versions files right away, but it could be good. 
    # For now, deleting the physical file and split is enough to remove it from the UI.
    
    return {"status": "success", "message": f"Deleted {deleted_count} images"}
