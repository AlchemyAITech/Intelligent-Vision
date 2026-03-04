from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import shutil
import json

router = APIRouter()

class CreateModelProjectRequest(BaseModel):
    project_name: str
    description: str = ""

class RenameModelProjectRequest(BaseModel):
    new_name: str

@router.get("/list")
async def list_model_projects():
    """获取所有已创建的模型工程仓列表"""
    projects_dir = os.path.join("data", "model_projects")
    os.makedirs(projects_dir, exist_ok=True)
    
    projects = []
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
                
            projects.append({
                "name": p,
                "description": description
            })
            
    return {"status": "success", "data": projects}

@router.post("/create")
async def create_model_project(req: CreateModelProjectRequest):
    """单独创建一个模型训练工程仓（不附带真实图片源数据）"""
    p_path = os.path.join("data", "model_projects", req.project_name)
    if os.path.exists(p_path):
        raise HTTPException(status_code=400, detail="同名工程仓已存在")
        
    os.makedirs(p_path, exist_ok=True)
    # 建立跑训练时的权重保存根节点
    os.makedirs(os.path.join(p_path, "runs"), exist_ok=True)
    
    # 保存元数据
    meta_path = os.path.join(p_path, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"description": req.description}, f, ensure_ascii=False)
        
    return {"status": "success", "message": f"成功创立算法工程仓: {req.project_name}"}

@router.delete("/{project_name}")
async def delete_model_project(project_name: str):
    """删除模型工程及其所有生成的权重及训练记录，不会触碰图片资产"""
    p_path = os.path.join("data", "model_projects", project_name)
    if not os.path.exists(p_path):
        raise HTTPException(status_code=404, detail="工程仓不存在")
        
    try:
        shutil.rmtree(p_path)
        return {"status": "success", "message": f"已彻底销毁工程仓及所有其附加实验权重: {project_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@router.post("/rename/{project_name}")
async def rename_model_project(project_name: str, req: RenameModelProjectRequest):
    """修改工程仓名称"""
    p_path = os.path.join("data", "model_projects", project_name)
    new_p_path = os.path.join("data", "model_projects", req.new_name)
    
    if not os.path.exists(p_path):
        raise HTTPException(status_code=404, detail="工程仓不存在")
    if os.path.exists(new_p_path):
        raise HTTPException(status_code=400, detail="同名工程仓已存在")
        
    try:
        os.rename(p_path, new_p_path)
        return {"status": "success", "message": f"成功重命名为: {req.new_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重命名失败: {str(e)}")
