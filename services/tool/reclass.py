import os
import yaml

# ==== CONFIGURATION ====
data_yaml_path = 'hcmus_thesis_2/data.yaml'  # Path to YAML file
label_dirs = ['hcmus_thesis_2/train/labels','hcmus_thesis_2/valid/labels']  # YOLO label directories

# ==== STEP 1: Read old_class_order from YAML ====
with open(data_yaml_path, 'r') as f:
    data = yaml.safe_load(f)
old_class_order = data['names']
print(f"[INFO] Loaded {len(old_class_order)} classes from {data_yaml_path}")

# ==== STEP 2: Define new_class_order as desired ====
new_class_order = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck',
    'other-vehicle',
    'traffic light',
    'stop sign',
    'Speed limit',
    'Speed limit 20km-h',
    'Speed limit 30km-h',
    'speed limit 40km-h',
    'Speed limit 50km-h',
    'Speed limit 60km-h',
    'Speed limit 70km-h',
    'Speed limit 80km-h',
    'Speed limit 100km-h',
    'Speed limit 120km-h',
    'End of speed limit 80km-h'
]
print(f"[INFO] Defined new class order with {len(new_class_order)} classes.")

# ==== STEP 3: Create mapping from old ID to new ID ====
old_to_new = {}
for old_id, class_name in enumerate(old_class_order):
    if class_name in new_class_order:
        new_id = new_class_order.index(class_name)
        old_to_new[old_id] = new_id
        print(f"[MAP] '{class_name}': {old_id} -> {new_id}")
    else:
        print(f"[WARNING] Class '{class_name}' in old list not found in new_class_order – skipping.")

# ==== STEP 4: Function to update label file ====
def update_label_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"[WARNING] Skipping malformed line in {filepath}: {line.strip()}")
            continue
        old_id = int(parts[0])
        if old_id in old_to_new:
            parts[0] = str(old_to_new[old_id])
            new_lines.append(' '.join(parts))
        else:
            print(f"[WARNING] Unknown class ID {old_id} in file {filepath} – skipping line.")

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"[UPDATED] {filepath}")

# ==== STEP 5: Apply to all label files ====
for label_dir in label_dirs:
    print(f"[INFO] Processing directory: {label_dir}")
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            update_label_file(os.path.join(label_dir, filename))

print("[DONE] All labels updated to new class order.")
