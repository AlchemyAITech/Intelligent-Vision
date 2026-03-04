import sys
from ultralytics import YOLO

def main():
    try:
        model = YOLO('yolov8n-cls.pt')
        results = model.train(data='/Users/yxh/workspace/MP项目/北京 - 清华大学/2026通识课/备课/第一节素材/Intelligent-Vision/data/projects/tiny-imagenet/yolo_train_dir', epochs=1, imgsz=224, batch=16, project='test_run', exist_ok=True)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
