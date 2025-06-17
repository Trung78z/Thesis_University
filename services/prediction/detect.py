# Example usage
import cv2
from AdvancedLaneDetector import AdvancedLaneDetector


if __name__ == "__main__":
    detector = AdvancedLaneDetector(yolo_model_path="models/t1.engine",)
    cap = cv2.VideoCapture("tool/video/test_video.mp4")
    
    while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            if processed_frame is not None:
                cv2.imshow('Lane and Car Detection', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()