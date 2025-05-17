from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Export to TensorRT engine
model.export(format="engine")  # Optional: device=0, half=True
