from ultralytics import YOLO
import cv2
import numpy as np
import time
from LaneDetector import LaneDetector
# Load the TensorRT-optimized model (.engine)
model = YOLO("runs/detect/train/weights/best.engine")

# Open video file or webcam (0 = default webcam)
video_path = "tool/output.mp4"
cap = cv2.VideoCapture(video_path)
fps = 0
frame_count = 0
start_time = time.time()
# Optional: Save the output to a new video
video_config = {
        'canny_thresholds': (70, 170),
        'hough_params': {
            'threshold': 50,
            'min_line_length': 30
        },
        'y_bottom_ratio': 0.95,
        'y_top_ratio': 0.75,
        'slope_threshold': 0.4,
        'x1': 200,
        'x2': 1200,
        'x3': 550,
        'x4': 680,
        'y':400
    }

detector = LaneDetector(video_config)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    frame = cv2.resize(frame,(1280,720))
    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS every 10 frames
        fps = 10 / (time.time() - start_time)
        start_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    results = model(frame)

    # Visualize results on frame
    annotated_frame = results[0].plot()


    annotated_frame, _ = detector.process_image(annotated_frame)
    # Show the frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
