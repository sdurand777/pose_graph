#!/usr/bin/env python3
"""
Script to copy images listed in id_kf_2_idx_img.txt from source to destination.
"""
import shutil
import ast
from pathlib import Path

# Paths
source_dir = Path("/home/ivm/escargot/imgs_left")
dest_dir = Path("/home/ivm/pose_graph/pgSlam/scenario/imgs")
mapping_file = Path("/home/ivm/pose_graph/pgSlam/scenario/id_kf_2_idx_img.txt")

# Read the list of image filenames
with open(mapping_file, 'r') as f:
    content = f.read().strip()
    # Parse the list (it's a Python list literal)
    image_list = ast.literal_eval(content)

print(f"Found {len(image_list)} images to copy")

# Create destination directory if it doesn't exist
dest_dir.mkdir(parents=True, exist_ok=True)

# Copy each image
copied = 0
errors = 0
for img_name in image_list:
    src_path = source_dir / img_name
    dst_path = dest_dir / img_name

    try:
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1
            if copied % 100 == 0:
                print(f"Copied {copied}/{len(image_list)} images...")
        else:
            print(f"Warning: {img_name} not found in source directory")
            errors += 1
    except Exception as e:
        print(f"Error copying {img_name}: {e}")
        errors += 1

print(f"\nCopy complete!")
print(f"Successfully copied: {copied}/{len(image_list)} images")
if errors > 0:
    print(f"Errors: {errors}")
