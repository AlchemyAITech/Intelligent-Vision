import torch
from ultralytics import YOLO
import asyncio
import os
import time

class UltralyticsEngine:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.active_jobs = {} # Tracking running training jobs

    def _get_optimal_device(self):
        """
        探测物理机硬件，实施 Apple MPS / CUDA 加速指令集的最优绑定。
        """
        if torch.cuda.is_available():
            print(">> [UltralyticsEngine] 侦测到 NVIDIA CUDA，将使用 cuda:0 满载运行")
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(">> [UltralyticsEngine] 🍎 侦测到 Apple Silicon (MPS)，异构加速引力矩阵已启动")
            return "mps"
        else:
            print(">> [UltralyticsEngine] 未侦测到加速集群，回退至原生 CPU 计算模式")
            return "cpu"

    async def train_model(
        self,
        job_id: str,
        project_name: str,
        yaml_path: str,
        model_type: str = "yolov8n",
        epochs: int = 10,
        batch_size: int = 8,
        optimizer: str = "auto",
        callback_ws=None
    ):
        """
        发起一次大模型/小型化模型挂载训练
        使用 Ultralytics 的 Callback 钩子 (callbacks="on_fit_epoch_end") 将指标推给 WebSocket。
        """
        proj_dir = os.path.join("data", "projects", project_name)
        run_dir = os.path.join(proj_dir, "runs", job_id)
        os.makedirs(run_dir, exist_ok=True)

        print(f">> [Training] 启动 {project_name} - {job_id} on {self.device} (Epochs: {epochs})")
        
        # 加载预训练底座
        model = YOLO(f"{model_type}.pt")

        # 挂载回调监控以注入 WS 数据流
        def on_train_epoch_end(trainer):
            # trainer.metrics 包含了 loss, map 等
            if callback_ws:
                # 提取当前 epoch 的损失和精度
                try:
                    metrics = trainer.metrics
                    metrics_payload = {
                        "epoch": trainer.epoch,
                        "box_loss": float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                        "cls_loss": float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                        "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
                        "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0))
                    }
                    # 异步推送，需要通过事件循环
                    asyncio.run_coroutine_threadsafe(
                        callback_ws.send_json(metrics_payload), 
                        asyncio.get_running_loop()
                    )
                except Exception as e:
                    print(f"[Engine Callback Error] {e}")

        # 注册回调 (覆盖 on_train_epoch_end 以防阻塞)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        self.active_jobs[job_id] = {"status": "running"}

        try:
            # 开始堵塞式训练。由于是 IO/GPU 密集型，生产中建议采用 ProcessPoolExecutor。
            # 这里出于演示和直接回调便利性，在外部线程调用。
            results = await asyncio.to_thread(
                model.train,
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                optimizer=optimizer,
                device=self.device,
                project=run_dir,
                name="train_session",
                exist_ok=True,
                verbose=False
            )
            self.active_jobs[job_id]["status"] = "success"
        except Exception as e:
            self.active_jobs[job_id]["status"] = f"failed: {str(e)}"
            raise e

    def export_onnx(self, project_name: str, job_id: str):
        """将 .pt 导出为跨平台的 ONNX"""
        pt_path = os.path.join("data", "projects", project_name, "runs", job_id, "train_session", "weights", "best.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"最佳权重不存在: {pt_path}")
        
        model = YOLO(pt_path)
        # format='onnx' 将触发后端转换
        exported_path = model.export(format="onnx", device=self.device)
        return exported_path

engine_instance = UltralyticsEngine()
