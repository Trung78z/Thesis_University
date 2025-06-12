from pathlib import Path

# Root directory containing the train/valid/test folders
base_dir = Path("dataVideo")
splits = ["train"]

remap_dict = {
    5: 4,  # speed_limit_40  -> speed limit (40km/h)
    7: 5,  # speed_limit_50  -> Speed limit (50km/h)
    9: 6,  # speed_limit_60  -> Speed limit (60km/h)
    11: 7   # speed_limit_80  -> Speed limit (80km/h)
}




# Temporary mapping to avoid overwriting old classes
temp_offset = 1000
temp_remap = {k: v + temp_offset for k, v in remap_dict.items()}

# Step 1: Change to temporary class IDs
print("Starting Step 1: Remap to temporary class IDs")
for split in splits:
    label_dir = base_dir / split / "labels"
    print(f"Processing split: {split}")
    for file in label_dir.glob("*.txt"):
        print(f"  Remapping file (to temp): {file}")
        new_lines = []
        with open(file) as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                if cls_id in temp_remap:
                    parts[0] = str(temp_remap[cls_id])
                new_lines.append(" ".join(parts))
        with open(file, "w") as f:
            f.write("\n".join(new_lines))
print("Step 1 complete.\n")

# Step 2: Change from temporary classes to destination classes
print("Starting Step 2: Remap to final class IDs")
for split in splits:
    label_dir = base_dir / split / "labels"
    print(f"Processing split: {split}")
    for file in label_dir.glob("*.txt"):
        print(f"  Remapping file (to final): {file}")
        new_lines = []
        with open(file) as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                if cls_id >= temp_offset:
                    for orig, temp in temp_remap.items():
                        if cls_id == temp:
                            parts[0] = str(remap_dict[orig])
                new_lines.append(" ".join(parts))
        with open(file, "w") as f:
            f.write("\n".join(new_lines))
print("Step 2 complete. Remapping finished.")