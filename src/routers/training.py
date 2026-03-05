from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import subprocess
import time
import sys
from threading import Thread

router = APIRouter()

class TrainClassifyRequest(BaseModel):
    project_name: str  # Now refers to the Model Project container
    dataset_name: str  # Refers to the source dataset
    version: str
    run_name: str = "run_default"
    model: str = "yolov8n-cls.pt"
    epochs: int = 10
    batch_size: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    patience: int = 50
    optimizer: str = "auto"
    augment: bool = False
    seed: int = 0

class StopTrainingRequest(BaseModel):
    project_name: str
    run_name: str

# Keyed by f"{project_name}_{run_name}"
ACTIVE_TRAINING = {}

def find_image_path(project_name: str, image_name: str):
    import json
    ds_dir = os.path.join("data", "projects", project_name, "dataset")
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
    for root, _, files in os.walk(ds_dir):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

import shutil

def run_yolo_train(project_name: str, dataset_name: str, version: str, run_name: str, model: str, epochs: int, batch_size: int, imgsz: int, lr0: float, patience: int, optimizer: str, augment: bool, seed: int):
    # 1. Prepare Paths for Model Project Output
    model_proj_dir = os.path.join("data", "model_projects", project_name)
    os.makedirs(model_proj_dir, exist_ok=True)
    run_dir = os.path.join(model_proj_dir, "runs", "classify", run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "training_logs.txt")
    job_key = f"{project_name}_{run_name}"
    
    # We write empty log file immediately so frontend can fetch (UTF-8 for consistent reading)
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Starting Training for Project: {project_name} | Dataset: {dataset_name} ({version}) | Run: {run_name}...\n")
        f.write(f"Model: {model}, Epochs: {epochs}, Batch: {batch_size}, Imgsz: {imgsz}\n")
    
    # 2. Re-create dataset structure for YOLOv8 Classification in the Model Project Workspace
    yolo_train_dir = os.path.join(model_proj_dir, "yolo_train_dir")
    
    if os.path.exists(yolo_train_dir):
        shutil.rmtree(yolo_train_dir)
    os.makedirs(yolo_train_dir, exist_ok=True)
    
    # Resolve the physical source dataset metadata
    dataset_dir = os.path.join("data", "projects", dataset_name)
    import json
    splits_file = os.path.join(dataset_dir, "splits.json")
    ann_file = os.path.join(dataset_dir, "annotations", f"{version}.json")
    
    splits_map = {}
    if os.path.exists(splits_file):
        with open(splits_file, "r", encoding="utf-8") as f:
            splits_map = json.load(f)
            
    ann_map = {}
    if os.path.exists(ann_file):
        with open(ann_file, "r", encoding="utf-8") as f:
            ann_map = json.load(f)
            
    copied_count = 0
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write("Preprocessing dataset distribution...\n")
        
    for img_name, split in splits_map.items():
        if split not in ["train", "val", "test"]: continue
        tag = ann_map.get(img_name)
        if not tag or tag in ["_description", "images"]: continue
        
        src_path = find_image_path(dataset_name, img_name)
        if src_path:
            dst_dir = os.path.join(yolo_train_dir, split, tag)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, img_name)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"Dataset Data Dir: {os.path.abspath(yolo_train_dir)} (Copied {copied_count} images)\n")
        f.write("-" * 50 + "\n")
        
    yolo_bin = os.path.join(os.path.dirname(sys.executable), "yolo")
    cmd = [
        yolo_bin, "classify", "train",
        f"data={os.path.abspath(yolo_train_dir)}",
        f"model={model}",
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"imgsz={imgsz}",
        f"lr0={lr0}",
        f"patience={patience}",
        f"optimizer={optimizer}",
        f"augment={augment}",
        f"seed={seed}",
        f"project={os.path.abspath(model_proj_dir)}",
        f"name=runs/classify/{run_name}",
        "exist_ok=True"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        
        ACTIVE_TRAINING[job_key] = process
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            for line in iter(process.stdout.readline, ''):
                f.write(line)
                f.flush()
        
        process.wait()
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 50 + "\n")
            if process.returncode == 0:
                f.write("Training Completed Successfully!\n")
            else:
                f.write(f"Training Failed with exit code {process.returncode}\n")
                
    except Exception as e:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\nError: {str(e)}\n")
    finally:
        ACTIVE_TRAINING.pop(job_key, None)

@router.post("/classify/start")
async def start_training(req: TrainClassifyRequest, background_tasks: BackgroundTasks):
    proj_dir = os.path.join("data", "model_projects", req.project_name)
    if not os.path.exists(proj_dir):
        raise HTTPException(status_code=404, detail="Model Project workspace not found")
        
    job_key = f"{req.project_name}_{req.run_name}"
    if job_key in ACTIVE_TRAINING:
        raise HTTPException(status_code=400, detail="Training is already running for this project and run name")
        
    # YOLO classification uses absolute dir containing train/val subdirectories
    # This check is now against the source dataset, not the model project
    source_dataset_path = os.path.join("data", "projects", req.dataset_name)
    dataset_images_dir = os.path.join(source_dataset_path, "dataset", "images")
    if not os.path.exists(os.path.join(dataset_images_dir, "train")):
        raise HTTPException(status_code=400, detail="Missing 'train' folder in source dataset")
        
    # Using background task to not block API
    thread = Thread(target=run_yolo_train, args=(
            req.project_name, 
            req.dataset_name,
            req.version, 
            req.run_name, 
            req.model, 
            req.epochs, 
            req.batch_size, 
            req.imgsz,
            req.lr0,
            req.patience,
            req.optimizer,
            req.augment,
            req.seed
        ))
    thread.daemon = True
    thread.start()
    
    # Slightly sleep to allow log file initialization for immediate polling
    time.sleep(0.5)
    
    return {"status": "success", "message": "Training started in background", "run_name": req.run_name}

@router.get("/classify/active")
async def check_active_training(project_name: str):
    """Check if any training daemon is actively running for this Model Project."""
    prefix = f"{project_name}_"
    active_runs = []
    
    # Iterate and clean up any dead threads
    keys_to_remove = []
    for key, thread in ACTIVE_TRAINING.items():
        if key.startswith(prefix):
            if thread.is_alive():
                run_name = key[len(prefix):]
                active_runs.append(run_name)
            else:
                keys_to_remove.append(key)
                
    for key in keys_to_remove:
        ACTIVE_TRAINING.pop(key, None)
        
    if len(active_runs) > 0:
        return {"status": "success", "is_running": True, "active_run": active_runs[0]}
    
    return {"status": "success", "is_running": False}

@router.post("/classify/stop")
async def stop_training(req: StopTrainingRequest):
    """Forcefully terminate an active training thread without deleting output directory."""
    job_key = f"{req.project_name}_{req.run_name}"
    
    # Actually, ACTIVE_TRAINING is storing Process objects (multiprocessing.Process), not threads
    # run_yolo_train is called natively, wait, line 164 says thread = Thread(...), 
    # Python Threads cannot be forcefully killed so easily if they are stuck in C-extensions.
    # We might need to rethink killing if it's a python threading.Thread.
    # Ah, the YOLO CLI call is inside run_yolo_train via os.system("yolo ...") which is synchronous.
    # Actually, we can use `pgrep -f "yolo.*name=runs/classify/{req.run_name}"` and kill it.
    
    # To reliably kill the YOLO subprocess launched by the thread:
    import subprocess
    target_name = f"name=runs/classify/{req.run_name}"
    try:
        # Find all pids containing the run_name target
        ps = subprocess.check_output(["pgrep", "-f", target_name]).decode('utf-8').split()
        for pid in ps:
            subprocess.run(["kill", "-9", pid])
            
        ACTIVE_TRAINING.pop(job_key, None)
        return {"status": "success", "message": "Training process terminated"}
    except subprocess.CalledProcessError:
        ACTIVE_TRAINING.pop(job_key, None)
        return {"status": "success", "message": "No active process found to kill, but cleared from registry"}

@router.get("/classify/logs")
async def get_training_logs(project_name: str, run_name: str = "run_default", lines: int = 100):
    # Search under model_projects
    log_file_path = os.path.join("data", "model_projects", project_name, "runs", "classify", run_name, "training_logs.txt")
    job_key = f"{project_name}_{run_name}"
    
    if not os.path.exists(log_file_path):
        return {"logs": "No training logs found.", "is_running": False}

    def _read_log_file(path: str, encodings: list[str] = None):
        if encodings is None:
            encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.readlines()
            except (UnicodeDecodeError, LookupError):
                continue
        # 最后兜底：用 utf-8 并替换无法解码的字节
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()

    try:
        content = _read_log_file(log_file_path)
        tail = content[-lines:] if len(content) > lines else content
        return {
            "logs": "".join(tail),
            "is_running": job_key in ACTIVE_TRAINING
        }
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}", "is_running": False}

@router.get("/runs")
async def list_runs(project_name: str):
    runs_dir = os.path.join("data", "model_projects", project_name, "runs", "classify")
    if not os.path.exists(runs_dir):
        return {"status": "success", "data": []}

    runs = []
    for d in os.listdir(runs_dir):
        dp = os.path.join(runs_dir, d)
        if os.path.isdir(dp):
            import datetime
            ctime = os.path.getctime(dp)
            cdate = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if weights exist
            weights_path = os.path.join(dp, "weights", "best.pt")
            has_weights = os.path.exists(weights_path)
            
            job_key = f"{project_name}_{d}"
            status = "running" if job_key in ACTIVE_TRAINING else ("completed" if has_weights else "failed/stopped")
            
            runs.append({
                "run_name": d,
                "created_at": cdate,
                "has_weights": has_weights,
                "status": status
            })
            
    # Sort by created_at descending
    runs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"status": "success", "data": runs}

@router.delete("/runs/{run_name}")
async def delete_run(project_name: str, run_name: str):
    job_key = f"{project_name}_{run_name}"
    if job_key in ACTIVE_TRAINING:
        # Terminate process if running
        proc = ACTIVE_TRAINING[job_key]
        proc.terminate()
        ACTIVE_TRAINING.pop(job_key, None)
        time.sleep(0.5)
        
    run_dir = os.path.join("data", "model_projects", project_name, "runs", "classify", run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        return {"status": "success", "message": f"Run {run_name} deleted."}
    else:
        raise HTTPException(status_code=404, detail="Run not found")

@router.get("/runs/{run_name}/details")
async def get_run_details(project_name: str, run_name: str):
    import yaml
    import csv
    
    run_dir = os.path.join("data", "model_projects", project_name, "runs", "classify", run_name)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")
        
    args_data = {}
    args_path = os.path.join(run_dir, "args.yaml")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r", encoding="utf-8") as f:
                args_data = yaml.safe_load(f)
        except Exception as e:
            pass # fallback to empty dict
            
    results_data = []
    results_path = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Strip spaces from keys and values
                for row in reader:
                    clean_row = {k.strip(): v.strip() for k, v in row.items()}
                    results_data.append(clean_row)
        except Exception as e:
            pass

    return {
        "status": "success",
        "data": {
            "args": args_data,
            "results": results_data
        }
    }

@router.get("/stats/{project_name}")
async def get_project_stats(project_name: str):
    """Get the running history statistics for a particular model project"""
    runs_dir = os.path.join("data", "model_projects", project_name, "runs", "classify")
    run_count = 0
    if os.path.exists(runs_dir):
        # count directories
        for entry in os.listdir(runs_dir):
            if os.path.isdir(os.path.join(runs_dir, entry)):
                run_count += 1
                
    return {
        "status": "success", 
        "data": {
            "run_count": run_count
        }
    }
