from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Test it on an image
results = model("data/test.jpeg")

# Show result
results[0].show()


onnx_model = YOLO("runs/detect/train/weights/best.onnx")
results = model("data/test.jpeg")
results[0].show()


onnx_model = YOLO("runs/detect/train/weights/best.engine")
results = model("data/test.jpeg")
results[0].show()
