from ultralytics import YOLO
import os
from glob import glob

model = YOLO('yolo11x.pt')  # or 'best.pt' if you have a trained model

image_folder = 'carvideo/train/images/'
output_folder = 'carvideo/train/labels/'
os.makedirs(output_folder, exist_ok=True)

image_paths = glob(os.path.join(image_folder, '*.*'))

for image_path in image_paths:
    results = model(image_path, conf=0.4, iou=0.5)  # set confidence and IoU thresholds

    for result in results:
        # Filter out overlapping boxes (NMS is already applied, but you can filter further if needed)
        boxes = result.boxes
        # Optionally, filter by confidence
        filtered_boxes = [box for box in boxes if box.conf > 0.4]

        txt_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_name)

        with open(txt_path, "w") as f:
            for box in filtered_boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                # Write in YOLO format: class x_center y_center width height
                f.write(f"{cls} {' '.join(map(str, xywh))}\n")
        print(f"Annotated: {os.path.basename(image_path)} -> {txt_path}")
