from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/detect/weights/best.pt")

# Export to TensorRT engine
model.export(format="onnx",device=0)  # Optional: device=0, half=True
model.export(format="engine",device=0)  # Optional: device=0, half=True
