import os

# Cấu hình
subset_dir = 'dataset/obj_train_data'  # hoặc 'obj_val_data'
output_file = 'dataset/train.txt'

# Lọc các file ảnh
image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

image_files = [
    os.path.join(subset_dir, f)
    for f in sorted(os.listdir(subset_dir))
    if f.lower().endswith(image_exts)
]

# Ghi ra file
with open(output_file, 'w') as f:
    for img in image_files:
        f.write(f"{img}\n")

print(f"{len(image_files)} image paths written to {output_file}")
