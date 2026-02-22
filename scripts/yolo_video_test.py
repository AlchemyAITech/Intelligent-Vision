import os
import cv2
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # 使用基础的 YOLOv8n 模型
    
    video_path = "/Users/yxh/workspace/MP项目/北京 - 清华大学/2026通识课/备课/第一节素材/codev2/video/demo_video_2.mp4"
    output_path = "/Users/yxh/workspace/MP项目/北京 - 清华大学/2026通识课/备课/第一节素材/codev2/video/demo_video_2_yolo_result.mp4"
    
    print(f"Start processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 进行推理，抑制详细日志
        results = model.predict(frame, verbose=False)
        
        # 绘制检测框和标签
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Success! YOLO detection Output saved to: {output_path}")

if __name__ == '__main__':
    main()
