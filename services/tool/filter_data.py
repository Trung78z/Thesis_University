from pathlib import Path
import shutil
from tqdm import tqdm

# COCO root path
coco_root = Path("./dataset/datas")

# Original label and image directories
labels_dir = coco_root / "train" / "labels"
images_dir = coco_root / "train" / "images"

# Output directories for filtered labels and images
filter_root = Path("data")
filter_root.mkdir(exist_ok=True)
filtered_images_dir = filter_root / "train" / "images"
filtered_images_dir.mkdir(parents=True, exist_ok=True)
filtered_labels_dir = filter_root / "train" / "labels"
filtered_labels_dir.mkdir(parents=True, exist_ok=True)
filtered_labels_dir.mkdir(exist_ok=True)
filtered_images_dir.mkdir(exist_ok=True)

# COCO class IDs to keep for ACC
# Original COCO class IDs: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
keep_ids = {0,1,2,3,5,7,9,11}  # Keeping all classes for ACC

# Start filtering
print("🔍 Filtering labels and images related to ACC...")
for label_path in tqdm(sorted(labels_dir.glob("*.txt"))):
    with label_path.open("r") as f:
        lines = f.readlines()

    kept = [line for line in lines if int(line.split()[0]) in keep_ids]

    if kept:
        # Write filtered labels
        (filtered_labels_dir / label_path.name).write_text("".join(kept))

        # Copy corresponding image
        img_path_jpg = images_dir / (label_path.stem + ".jpg")
        img_path_png = images_dir / (label_path.stem + ".png")
        img_path_jpeg = images_dir / (label_path.stem + ".jpeg")

        if img_path_jpg.exists():
            shutil.copy(img_path_jpg, filtered_images_dir / img_path_jpg.name)
        elif img_path_png.exists():
            shutil.copy(img_path_png, filtered_images_dir / img_path_png.name)
        elif img_path_jpeg.exists():
            shutil.copy(img_path_jpeg, filtered_images_dir / img_path_jpeg.name)

print("✅ Filtering complete. Filtered data located at:")
print(f"  Labels: {filtered_labels_dir}")
print(f"  Images: {filtered_images_dir}")
