from ultralytics import YOLO
import cv2

# Load model
model = YOLO("../runs/detect/train3/weights/best.pt")

# Load and resize image
# image_path = "data/train/images/9567fb1e-car_0007.jpg"

image_path = "data/train/images/92f4749e-car_0006.jpg"

# Run detection
results = model(image_path)

# Show result
results[0].show()
