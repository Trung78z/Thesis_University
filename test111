import cv2
import time
from ultralytics import solutions
video_path = "tool/video.mp4"
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
frame_count = 0
start_time = time.time()
# Initialize SpeedEstimator
speedestimator = solutions.SpeedEstimator(
    model="runs/detect/train/weights/best.engine",
    show=True,
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS every 10 frames
        fps = 10 / (time.time() - start_time)
        start_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    results = speedestimator(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()