#!/usr/bin/env python3
"""
Script to rename images with zero-padding to 5 digits (img00000.jpg to img01197.jpg)
"""
from pathlib import Path

# Paths
dest_dir = Path("/home/ivm/pose_graph/pgSlam/scenario/imgs")

# Get all current images and sort them numerically
images = sorted(dest_dir.glob("img*.jpg"), key=lambda x: int(x.stem[3:]))

print(f"Found {len(images)} images to rename with zero-padding")

# Rename each image with zero-padding
renamed = 0
for img_path in images:
    # Extract the current number
    current_num = int(img_path.stem[3:])

    # Create new name with 5-digit zero-padding
    new_name = f"img{current_num:05d}.jpg"
    new_path = dest_dir / new_name

    try:
        if img_path != new_path:  # Only rename if different
            img_path.rename(new_path)
            renamed += 1
            if renamed % 100 == 0:
                print(f"Renamed {renamed}/{len(images)} images...")
        else:
            print(f"Skipping {img_path.name} (already has correct format)")
    except Exception as e:
        print(f"Error renaming {img_path.name}: {e}")

print(f"\nRename complete!")
print(f"Successfully renamed: {renamed}/{len(images)} images")
print(f"Images now formatted as img00000.jpg to img{len(images)-1:05d}.jpg")
