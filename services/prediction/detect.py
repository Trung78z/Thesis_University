# Example usage
import cv2
from AdvancedLaneDetector import AdvancedLaneDetector


if __name__ == "__main__":
    detector = AdvancedLaneDetector(yolo_model_path="yolo11l-seg.pt")
    cap = cv2.VideoCapture("tool/trip_cut.mp4")
    
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