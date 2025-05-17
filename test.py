from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Test it on an image
results = model("27094_3063d356a3a54cc3859537fd23c5ba9d_1539205710.jpeg")

# Show result
results[0].show()


onnx_model = YOLO("runs/detect/train/weights/best.onnx")
results = model("27094_3063d356a3a54cc3859537fd23c5ba9d_1539205710.jpeg")
results[0].show()


onnx_model = YOLO("runs/detect/train/weights/best.engine")
results = model("27094_3063d356a3a54cc3859537fd23c5ba9d_1539205710.jpeg")
results[0].show()
