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

@router.post("/pca_cluster")
async def perform_pca_clustering(req: PCARequest):
    """
    通过 scikit-learn 对高维影像特征进行 PCA 主成分降维。
    在实际生产中，这里应当是提取大模型最后一层池化特征 (如 512维)，再将其压缩至 2 维抛给前端散点图。
    目前为了 MVP 联调前置化，这部分使用真实 sklearn 引擎加生成聚集特征簇进行演示。
    """
    n_samples = req.sample_count
    
    # 构建 3 个具有内在聚类趋势的高维 (128维) 特征空间
    # 模拟医学检验中的三种类别：良性、恶性、正常
    cluster_1 = np.random.normal(loc=0.5, scale=0.8, size=(n_samples // 3, 128))
    cluster_2 = np.random.normal(loc=-1.0, scale=0.5, size=(n_samples // 3, 128))
    cluster_3 = np.random.normal(loc=2.0, scale=0.6, size=(n_samples - (n_samples // 3) * 2, 128))
    
    X = np.vstack([cluster_1, cluster_2, cluster_3])
    labels = ['Benign'] * (n_samples // 3) + ['Malignant'] * (n_samples // 3) + ['Normal'] * (n_samples - (n_samples // 3) * 2)

    # 真实下发 scikit-learn 引擎跑降维计算
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # 包装回传给 ECharts
    scatter_data = []
    for i in range(len(labels)):
        scatter_data.append({
            "x": float(X_reduced[i, 0]),
            "y": float(X_reduced[i, 1]),
            "label": labels[i]
        })

    return {
        "status": "success",
        "variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "points": scatter_data
    }

@router.post("/grad_cam")
async def generate_grad_cam(project_name: str, file: UploadFile = File(...)):
    """
    接收用户上传的测试图片，加载特定模型并抽出其激活分布图。
    本 MVP 返回一个与原图尺寸吻合并叠加了伪彩热力图处理的分析图 URL。
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"status": "error", "message": "图片解码失败"}

    h, w, _ = image.shape
    
    # 建立一个基于高斯分布的模拟注意焦点，模拟网络关注区域
    heatmap = np.zeros((h, w), dtype=np.float32)
    center_x, center_y = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
    cv2.circle(heatmap, (center_x, center_y), min(h, w)//4, 1.0, -1)
    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 30)
    
    # 归一化并伪彩映射
    heatmap = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 与原图 Alpha 混合
    cam_result = cv2.addWeighted(image, 0.6, colormap, 0.4, 0)

    # 存储并返回静态路由可访问的地址
    out_dir = os.path.join("uploads", "cam_results")
    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"cam_{np.random.randint(1000, 9999)}.jpg"
    out_path = os.path.join(out_dir, out_filename)
    
    cv2.imwrite(out_path, cam_result)

    return {
        "status": "success",
        "cam_url": f"/uploads/cam_results/{out_filename}",
        "focus_coordinate": [center_x, center_y]
    }
