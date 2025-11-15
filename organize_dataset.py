import os
import shutil
from pathlib import Path

# === Define paths ===
base = Path("data/crops for classification")  # your current folder
target_base = Path("data/train")  # where the organized data will go

# Define expected classes for each crop
disease_map = {
    "corn": ["healthy", "common_rust", "gray_leaf_spot", "northern_leaf_blight"],
    "potato": ["healthy", "early_blight", "late_blight"],
    "rice": ["healthy", "brown_spot", "leaf_blast", "neck_blast"],
    "sugarcane": ["healthy", "bacterial_blight", "red_rot"],
    "wheat": ["healthy", "brown_rust", "yellow_rust"]
}

# === Create target directories ===
for crop, diseases in disease_map.items():
    for disease in diseases:
        folder = target_base / crop / disease
        folder.mkdir(parents=True, exist_ok=True)

# === Move files based on keywords ===
for crop, diseases in disease_map.items():
    crop_path = base / crop
    if not crop_path.exists():
        print(f"⚠️ Crop folder missing: {crop}")
        continue

    for img in crop_path.glob("*.jpg"):
        moved = False
        for disease in diseases:
            if disease.replace("_", " ") in img.name.lower() or disease in img.name.lower():
                dest = target_base / crop / disease / img.name
                shutil.move(str(img), str(dest))
                moved = True
                break
        if not moved:
            dest = target_base / crop / "healthy" / img.name
            shutil.move(str(img), str(dest))

print("\n✅ Dataset Organized Successfully!")
print("All images are now grouped by crop and disease inside data/train/")
