from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Export the model to ONNX format
success = model.export(format="onnx")
