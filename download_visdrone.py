from ultralytics import YOLO
import os

def download_visdrone():
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"Downloading VisDrone dataset to {data_dir}...")
    
    # We use a dummy model to trigger the dataset download via 'train' or 'val'.
    # Because 'yolo detect train data=VisDrone.yaml' handles verification and download automatically.
    # We set epochs=0 to avoid actual training, just trigger the download preparation.
    
    try:
        model = YOLO("yolo11n.pt")
        # Use absolute path for project to avoid creating runs/ in weird places
        # We just want to trigger the download.
        print("Note: This will verify/download VisDrone.yaml dataset.")
        print("If it's the first time, it might take a while.")
        
        # 'data' arg in train() triggers the check_dataset routine.
        # We need to ensure settings point to our 'data' dir if we want it there.
        # Ultralytics settings usually point to a global dir or next to the yaml.
        
        # Let's inspect where it goes. By default it might go to ../datasets/VisDrone
        # To force it to ./data, we might need to modify settings or symlink.
        
        from ultralytics import settings
        print(f"Current datasets_dir: {settings['datasets_dir']}")
        
        # Update settings to point to local data dir
        settings.update({'datasets_dir': data_dir})
        print(f"Updated datasets_dir to: {data_dir}")
        
        # Trigger download
        model.train(data='VisDrone.yaml', epochs=0, imgsz=32, plots=False)
        print("Download/Verification process completed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_visdrone()
