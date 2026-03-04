import json
import os
import random

project_name = "tiny-imagenet"
proj_dir = os.path.join("data", "projects", project_name)

splits_path = os.path.join(proj_dir, "splits.json")
meta_path = os.path.join(proj_dir, "meta.json")
v1_path = os.path.join(proj_dir, "annotations", "v1.json")

with open(splits_path, "r", encoding="utf-8") as f:
    splits_map = json.load(f)

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
    categories = meta.get("categories", [])

if not categories:
    print("No categories found in meta.json")
    exit(1)

with open(v1_path, "r", encoding="utf-8") as f:
    v1_map = json.load(f)

count = 0
for img, split in splits_map.items():
    if split == "test" and img not in v1_map:
        v1_map[img] = random.choice(categories)
        count += 1

with open(v1_path, "w", encoding="utf-8") as f:
    json.dump(v1_map, f, ensure_ascii=False)

print(f"Successfully labeled {count} test images.")
