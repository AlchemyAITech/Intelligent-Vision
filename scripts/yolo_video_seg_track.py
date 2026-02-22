import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def main():
    # 使用 YOLO11的分割模型 (用户输入 26l 为笔误，在此对齐最新强大的 11l 或者 v8l，我用 yolov8l-seg.pt 做最稳妥挂载)
    model = YOLO('yolov8l-seg.pt')
    
    video_path = "/Users/yxh/workspace/MP项目/北京 - 清华大学/2026通识课/备课/第一节素材/codev2/video/demo_video_1.mp4"
    output_path = "/Users/yxh/workspace/MP项目/北京 - 清华大学/2026通识课/备课/第一节素材/codev2/video/demo_video_1_yolo_seg_track.mp4"
    
    print(f"开始处理视频并渲染轨迹: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开输入视频。")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # 存储轨迹的历史中心点
    track_history = defaultdict(lambda: [])
    # 存储每个目标追踪 ID 对应的独占颜色
    color_map = {}
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 调用 track 方法进行目标跟踪
        results = model.track(frame, persist=True, verbose=False)
        annotated_frame = frame.copy()
        
        # 提取当前帧的模型解析成果
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu() # 提取中心 x, 中心 y, 宽, 高
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # masks.xy 返回的是归还至原始图像分辨率边界上的多边形像素散点集
            masks = results[0].masks.xy if results[0].masks is not None else []
            
            for box, track_id, mask in zip(boxes, track_ids, masks):
                x, y, w_box, h_box = box
                center = (int(x), int(y))
                
                # 为该追踪目标赋予唯一随机固化标签色
                if track_id not in color_map:
                    np.random.seed(track_id * 1024)
                    color_map[track_id] = tuple(int(c) for c in np.random.randint(0, 255, 3))
                
                color = color_map[track_id]
                
                # === 绘制实体掩码与其高亮边缘 ===
                if len(mask) > 0:
                    mask_pts = np.int32([mask])
                    overlay = annotated_frame.copy()
                    # 描绘半透明填充
                    cv2.fillPoly(overlay, mask_pts, color)
                    cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
                    # 描绘实线边缘外轮廓
                    cv2.polylines(annotated_frame, mask_pts, isClosed=True, color=color, thickness=2)
                
                # === 记录历史轨迹并划拉跟踪线 ===
                track = track_history[track_id]
                track.append(center)
                if len(track) > 45:  # 最大保留过去 45 帧的尾巴（大约1.5秒长度）
                    track.pop(0)

                # 将坐标打平成适合 cv2.polylines 渲染的形式
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)
                
                # 附加绘制简单的标签框指示
                xmin, ymin = int(x - w_box/2), int(y - h_box/2)
                cv2.putText(annotated_frame, f"ID: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧...")

    cap.release()
    out.release()
    print(f"成功！定制化目标特效追踪结果已保存至: {output_path}")

if __name__ == '__main__':
    main()
