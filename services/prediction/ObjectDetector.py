import cv2
import numpy as np
from scipy import stats
from ultralytics import YOLO
from collections import deque

class ObjectDetector:
    def __init__(self, yolo_model_path="yolov8n.pt"):
        try:
            self.yolo_model = YOLO(yolo_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
        
        self.target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            print("Warning: Empty frame received")
            return None
            
        try:
            frame = cv2.resize(frame, (1280, 720))
            
            try:
                yolo_results = self.yolo_model(frame, classes=self.target_classes, verbose=False)
            except Exception as e:
                print(f"YOLO detection error: {str(e)}")
                yolo_results = []
            
            for result in yolo_results:
                try:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        cls_id = int(box.cls)
                        
                        if conf > 0.2:
                            color = (0, 255, 0) if cls_id == 2 else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            # print(f"Detected {cls_id} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")
                            label = f"{self.yolo_model.names[cls_id]} {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception:
                    continue
            
            return frame
            
        except Exception as e:
            print(f"Object detection error: {str(e)}")
            return None