from ultralytics import YOLO
import cv2
import numpy as np
import time
# Load the TensorRT-optimized model (.engine)
model = YOLO("runs/detect/train/weights/best.onnx")

# Open video file or webcam (0 = default webcam)
video_path = "tool/video.mp4"
cap = cv2.VideoCapture(video_path)
fps = 0
frame_count = 0
start_time = time.time()
# Optional: Save the output to a new video

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    frame = cv2.resize(frame,(1280,720))
    results = model(frame)

    # Visualize results on frame
    annotated_frame = results[0].plot()
    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS every 10 frames
        fps = 10 / (time.time() - start_time)
        start_time = time.time()
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Show the frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
