"""
Reorganize data/train structure for disease-level classification
Creates a flat structure: data/train_diseases/{crop}_{disease}/
"""

import shutil
from pathlib import Path

# === Paths ===
SOURCE_DIR = Path("data/train")
TARGET_DIR = Path("data/train_diseases")

print("=" * 60)
print("🔄 Reorganizing Dataset for Disease Classification")
print("=" * 60)

# Create target directory
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Counter for copied files
total_files = 0

# === Process each crop folder ===
for crop_dir in SOURCE_DIR.iterdir():
    if not crop_dir.is_dir():
        continue
    
    crop_name = crop_dir.name
    print(f"\n📁 Processing crop: {crop_name}")
    
    # Process each disease folder
    for disease_dir in crop_dir.iterdir():
        if not disease_dir.is_dir():
            continue
        
        disease_name = disease_dir.name
        
        # Create target folder: {crop}_{disease}
        target_folder = TARGET_DIR / f"{crop_name}_{disease_name}"
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        image_files = list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.JPG")) + \
                     list(disease_dir.glob("*.jpeg")) + list(disease_dir.glob("*.JPEG")) + \
                     list(disease_dir.glob("*.png")) + list(disease_dir.glob("*.PNG"))
        
        copied = 0
        for img_file in image_files:
            try:
                dest = target_folder / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    copied += 1
            except Exception as e:
                print(f"   ⚠️ Error copying {img_file.name}: {e}")
        
        total_files += copied
        print(f"   ✅ {disease_name}: {copied} images → {target_folder.name}")

print("\n" + "=" * 60)
print(f"✅ Reorganization complete!")
print(f"   Total images copied: {total_files}")
print(f"   Target directory: {TARGET_DIR}")
print("=" * 60)

# List created classes
classes = sorted([d.name for d in TARGET_DIR.iterdir() if d.is_dir()])
print(f"\n📊 Created {len(classes)} disease classes:")
for i, cls in enumerate(classes, 1):
    print(f"   {i}. {cls}")

