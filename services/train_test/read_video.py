import cv2
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        frame = cv2.resize(frame, (1280, 720))  # Resize frame to 1280x720
        # Display the frame
        cv2.imshow('Video Frame', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'tool/test_video.mp4'  # Replace with your video file path
    read_video(video_path)