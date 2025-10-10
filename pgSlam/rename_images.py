#!/usr/bin/env python3
"""
Script to rename images sequentially from img0.jpg to img1197.jpg
"""
import ast
from pathlib import Path

# Paths
dest_dir = Path("/home/ivm/pose_graph/pgSlam/scenario/imgs")
mapping_file = Path("/home/ivm/pose_graph/pgSlam/scenario/id_kf_2_idx_img.txt")

# Read the list of image filenames in order
with open(mapping_file, 'r') as f:
    content = f.read().strip()
    image_list = ast.literal_eval(content)

print(f"Found {len(image_list)} images to rename")

# Rename each image sequentially
renamed = 0
for idx, old_name in enumerate(image_list):
    old_path = dest_dir / old_name
    new_name = f"img{idx}.jpg"
    new_path = dest_dir / new_name

    try:
        if old_path.exists():
            old_path.rename(new_path)
            renamed += 1
            if renamed % 100 == 0:
                print(f"Renamed {renamed}/{len(image_list)} images...")
        else:
            print(f"Warning: {old_name} not found")
    except Exception as e:
        print(f"Error renaming {old_name}: {e}")

print(f"\nRename complete!")
print(f"Successfully renamed: {renamed}/{len(image_list)} images")
print(f"Images now numbered from img0.jpg to img{len(image_list)-1}.jpg")
