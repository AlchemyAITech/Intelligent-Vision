import torch
import cv2
import os
import shutil
import sys

# Load backend API directly
from src.routers.sam import SAM3Backend

def test():
    print(f"Using device: mps if available, else cpu")
    backend = SAM3Backend()
    
    if not backend.video_predictor:
        print("Initializing video predictor...")
        from sum3_repo.sam3.build_sam import build_sam3_video_predictor
        # 强制在 MPS 上构建（和我们前端的调用逻辑一致）
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        backend.video_predictor = build_sam3_video_predictor(
            "sum3_repo/sam3_configs/sam3_1.yaml",
            "sum3_repo/weights/sam3.1_hiera_large.pt",
            device=device
        )

    # 制造一些假的测试图片序列，免除读长篇大论视频
    print("Creating mock frames...")
    os.makedirs("temp_test_frames", exist_ok=True)
    for i in range(3):
        # 创建空白图片模拟
        img = torch.zeros(300, 300, 3, dtype=torch.uint8).numpy()
        cv2.imwrite(f"temp_test_frames/{i:05d}.jpg", img)

    print("Start session...")
    res_start = backend.video_predictor.start_session(resource_path="temp_test_frames", session_id="test_sess")
    session_id = res_start["session_id"]

    print("Adding dummy box/point prompt...")
    backend.video_predictor.add_prompt(
        session_id=session_id,
        frame_idx=0,
        obj_id=1,
        points=[[150, 150]],
        point_labels=[1],
        bounding_boxes=None,
        bounding_box_labels=None,
        text=None
    )

    print("Running propagate_in_video()...")
    # 这就是用户点击所爆了三次错的最深的地方
    try:
        for out_dict in backend.video_predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="both",
            start_frame_idx=0,
            max_frame_num_to_track=None
        ):
            print(f" - Got frame prediction: {out_dict.get('frame_index')}")
        print("SUCCESS! ALL CLEAR.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR: Something went wrong again.")
    finally:
        shutil.rmtree("temp_test_frames")

if __name__ == "__main__":
    test()
