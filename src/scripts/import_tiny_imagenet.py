import os
import json
import shutil
import glob
from tqdm import tqdm

SOURCE_DIR = "data/tiny-imagenet-200"
DEST_PROJECT = "tiny-imagenet"
PROJECT_DIR = os.path.join("data", "projects", DEST_PROJECT)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

def main():
    print(f"Importing {SOURCE_DIR} into {PROJECT_DIR}...")
    
    # 1. Load wnids mapping
    words_file = os.path.join(SOURCE_DIR, "words.txt")
    wnid_to_desc = {}
    with open(words_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                wnid, desc = parts[0], parts[1].split(",")[0].strip() # Take first alias
                wnid_to_desc[wnid] = desc

    # 2. Setup destination directories
    for split in ["train", "val", "test", "unassigned"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "annotations"), exist_ok=True)
    
    # 3. Preparation for meta and splits
    splits_map = {}
    ann_map = {}
    categories = set()
    
    # helper for tracking
    def process_image(src_path, filename, split_name, category_name):
        dest_path = os.path.join(DATASET_DIR, "images", split_name, filename)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)
        splits_map[filename] = split_name
        if category_name:
            ann_map[filename] = category_name
            categories.add(category_name)
    
    # 4. Process train
    train_dir = os.path.join(SOURCE_DIR, "train")
    if os.path.exists(train_dir):
        wnids = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        print("Processing training images...")
        for wnid in tqdm(wnids):
            desc = wnid_to_desc.get(wnid, wnid)
            img_folder = os.path.join(train_dir, wnid, "images")
            if os.path.exists(img_folder):
                for img_name in os.listdir(img_folder):
                    if img_name.lower().endswith('.jpeg'):
                        src_path = os.path.join(img_folder, img_name)
                        process_image(src_path, img_name, "train", desc)
                        
    # 5. Process val
    val_dir = os.path.join(SOURCE_DIR, "val")
    val_ann_file = os.path.join(val_dir, "val_annotations.txt")
    val_images_dir = os.path.join(val_dir, "images")
    if os.path.exists(val_ann_file) and os.path.exists(val_images_dir):
        print("Processing validation images...")
        with open(val_ann_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_name, wnid = parts[0], parts[1]
                    desc = wnid_to_desc.get(wnid, wnid)
                    src_path = os.path.join(val_images_dir, img_name)
                    if os.path.exists(src_path):
                        process_image(src_path, img_name, "val", desc)
                        
    # 6. Process test
    test_images_dir = os.path.join(SOURCE_DIR, "test", "images")
    if os.path.exists(test_images_dir):
        print("Processing testing images...")
        test_imgs = [f for f in os.listdir(test_images_dir) if f.lower().endswith('.jpeg')]
        for img_name in tqdm(test_imgs):
            src_path = os.path.join(test_images_dir, img_name)
            process_image(src_path, img_name, "test", None)
            
    # 7. Save structural state
    with open(os.path.join(PROJECT_DIR, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits_map, f, ensure_ascii=False)
        
    v1_path = os.path.join(PROJECT_DIR, "annotations", "v1.json")
    with open(v1_path, "w", encoding="utf-8") as f:
        json.dump(ann_map, f, ensure_ascii=False)
        
    meta = {
        "description": "Tiny ImageNet Challenge Dataset",
        "categories": sorted(list(categories))
    }
    with open(os.path.join(PROJECT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    print(f"Import complete! Total images: {len(splits_map)}")
    print(f"Total categories: {len(categories)}")

if __name__ == "__main__":
    main()
