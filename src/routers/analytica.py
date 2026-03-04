from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
import numpy as np
import cv2
from typing import List
import os
from sklearn.decomposition import PCA

router = APIRouter()

class PCARequest(BaseModel):
    project_name: str
    sample_count: int = 200 # 模拟提取向量的数量

def get_latest_best_pt(project_name: str):
    runs_dir = os.path.join("data", "model_projects", project_name, "runs", "classify")
    if not os.path.exists(runs_dir):
        return None
    valid_runs = []
    for d in os.listdir(runs_dir):
        dp = os.path.join(runs_dir, d)
        if os.path.isdir(dp):
            pt_path = os.path.join(dp, "weights", "best.pt")
            if os.path.exists(pt_path):
                valid_runs.append((os.path.getmtime(pt_path), pt_path))
    if not valid_runs:
        return None
    valid_runs.sort(key=lambda x: x[0], reverse=True)
    return valid_runs[0][1]

@router.post("/pca_cluster")
async def perform_pca_clustering(req: PCARequest):
    """
    通过 scikit-learn 对高维影像特征进行真实 PCA 主成分降维。
    提取当前项目最新训练的 best.pt 大模型倒数第二层池化提取的 256维张量。
    """
    pt_path = get_latest_best_pt(req.project_name)
    if not pt_path:
         return {"status": "error", "message": "未找到在此工程下训练产生的可用权重 (best.pt)"}

    from ultralytics import YOLO
    import torch
    import cv2
    import random
    
    yolo_val_dir = os.path.join("data", "model_projects", req.project_name, "yolo_train_dir", "val")
    if not os.path.exists(yolo_val_dir):
         return {"status": "error", "message": "未找到模型配套的交叉验证集测试图像"}
         
    classes = [d for d in os.listdir(yolo_val_dir) if os.path.isdir(os.path.join(yolo_val_dir, d))]
    images_to_process = []
    
    for cls in classes:
        cls_dir = os.path.join(yolo_val_dir, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        # sample images capping roughly to total req limit
        per_class_limit = max(1, req.sample_count // max(1, len(classes)))
        if len(imgs) > per_class_limit:
            imgs = random.sample(imgs, per_class_limit)
        for img in imgs:
            images_to_process.append((img, cls))

    if not images_to_process:
        return {"status": "error", "message": "验证集没有测试图片支持流形计算"}

    try:
        model = YOLO(pt_path)
    except Exception as e:
        return {"status": "error", "message": f"加载模型权重失败: {str(e)}"}
        
    X_list = []
    labels_list = []
    
    # Run embed network inference
    for img_path, cls_label in images_to_process:
        try:
            res = model.embed(img_path, verbose=False)
            if isinstance(res, list) and len(res) > 0:
                tensor = res[0].cpu().numpy()
                X_list.append(tensor)
                labels_list.append(cls_label)
        except Exception:
            continue
            
    if len(X_list) < 3:
         return {"status": "error", "message": "特征提取向量簇太小，不足以进行拓扑空间降维"}
         
    X = np.vstack(X_list)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # 包裹给前台 Echarts
    scatter_data = []
    for i in range(len(labels_list)):
        scatter_data.append({
            "x": float(X_reduced[i, 0]),
            "y": float(X_reduced[i, 1]),
            "label": labels_list[i]
        })

    return {
        "status": "success",
        "variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "points": scatter_data
    }

@router.post("/grad_cam")
async def generate_grad_cam(project_name: str, file: UploadFile = File(...)):
    """
    接收用户上传的测试图片，加载本次实验对应模型，使用 PyTorch Hooks 抽出最终卷积层响应分布图。
    生成吻合原图结构的真实伪彩热力图 (CAM)。
    """
    pt_path = get_latest_best_pt(project_name)
    if not pt_path:
         return {"status": "error", "message": "未找到在此工程下训练产生的可用权重 (best.pt)"}

    # Load file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"status": "error", "message": "图片解码失败"}

    from ultralytics import YOLO
    import torch
    
    try:
        model = YOLO(pt_path)
    except Exception as e:
        return {"status": "error", "message": f"挂载安全模型错误: {str(e)}"}
        
    target_layer = model.model.model[-2] # Final Conv layer
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Create temp path for YOLO to read
    os.makedirs("uploads/cam_tmp", exist_ok=True)
    tmp_path = os.path.join("uploads/cam_tmp", f"cam_tmp_{np.random.randint(1000, 9999)}.jpg")
    cv2.imwrite(tmp_path, image)
    
    # Run forward pass
    res = model(tmp_path, verbose=False)
    handle.remove()
    
    if len(activations) == 0:
        return {"status": "error", "message": "反向网络未收集到特征图分布"}
        
    act = activations[0] # [1, C, H, W]
    # Simple Activation Mapping: average over channel dimension
    am = torch.mean(act, dim=1).squeeze().numpy() # [H, W]
    
    # Normalize to 0-255
    am = am - np.min(am)
    if np.max(am) != 0:
        am = am / np.max(am)
    heatmap = np.uint8(255 * am)
    
    # Resize and blend
    h, w, _ = image.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    cam_result = cv2.addWeighted(image, 0.5, colormap, 0.5, 0)

    out_dir = os.path.join("uploads", "cam_results")
    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"cam_{np.random.randint(100000, 999999)}.jpg"
    out_path = os.path.join(out_dir, out_filename)
    
    cv2.imwrite(out_path, cam_result)
    try:
        os.remove(tmp_path)
    except:
        pass

    return {
        "status": "success",
        "cam_url": f"/uploads/cam_results/{out_filename}"
    }
