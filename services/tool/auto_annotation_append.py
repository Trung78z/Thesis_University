#!/usr/bin/env python3
"""
Infer images with a YOLO model and append new annotations
for classes 10 – 19 (speed‑limit signs) without mất dữ liệu cũ.
"""

from pathlib import Path
from typing import List

from ultralytics import YOLO


# ----------------------------- CONFIG -------------------------------- #

CLASS_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "other-vehicle",
    "traffic light", "stop sign", "Speed limit", "Speed limit 20km-h",
    "Speed limit 30km-h", "speed limit 40km-h", "Speed limit 50km-h",
    "Speed limit 60km-h", "Speed limit 70km-h", "Speed limit 80km-h",
    "Speed limit 100km-h", "Speed limit 120km-h", "End of speed limit 80km-h"
]

MODEL_PATH   = Path("../models/traffic-sign-x.engine")
IMAGE_DIR    = Path("dataset/images")
LABEL_DIR    = Path("dataset/labels")
CONF_THRESH  = 0.40
IOU_THRESH   = 0.50
TARGET_RANGE = range(10, 20)          # classes 10 – 19 inclusive
# ---------------------------------------------------------------------- #


def load_model(model_path: Path) -> YOLO:
    """Load YOLO engine / weights."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def xywh_normalized(box_xywh, img_w: int, img_h: int) -> str:
    """Return a normalized xywh string ready to write to txt."""
    x, y, w, h = box_xywh
    return f"{x / img_w:.6f} {y / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def append_annotations(txt_path: Path, new_lines: List[str]) -> None:
    """Append new annotation lines to an existing YOLO‑format file (or create it)."""
    existing = txt_path.read_text().splitlines(keepends=True) if txt_path.exists() else []
    txt_path.write_text("".join(existing + new_lines))


def process_image(model: YOLO, image_path: Path) -> None:
    """Run inference on one image and update its label file."""
    results = model(str(image_path), conf=CONF_THRESH, iou=IOU_THRESH)

    for result in results:
        img_h, img_w = result.orig_shape
        new_lines: List[str] = []

        for box in result.boxes:
            if box.conf < CONF_THRESH:
                continue
            cls_id = int(box.cls)
            if cls_id not in TARGET_RANGE:
                continue
            xywh = box.xywh[0].tolist()
            line = f"{cls_id} {xywh_normalized(xywh, img_w, img_h)}\n"
            new_lines.append(line)
            print(f"Add {CLASS_NAMES[cls_id]:<25} → {image_path.stem}.txt")

        if new_lines:
            label_path = LABEL_DIR / f"{image_path.stem}.txt"
            append_annotations(label_path, new_lines)


def main() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL_PATH)

    images = sorted(IMAGE_DIR.glob("*.*"))
    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    for img in images:
        process_image(model, img)
        print(f"Processed {img.name}")


if __name__ == "__main__":
    main()
