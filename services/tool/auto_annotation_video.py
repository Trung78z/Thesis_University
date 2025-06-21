import cv2
import os
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

class CustomYOLOv11VideoProcessor:
    def __init__(self, model_path='yolo11x.pt'):
        """Initialize YOLOv11 model with custom class names."""
        self.model = YOLO(model_path)
        self.custom_classes = [
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'bus',
            'truck',
            'stop sign',
            'other-vehicle',
            'crosswalk',
            'red light',
            'yellow light',
            'green light',
            'Speed limit 30km-h',
            'Speed limit 40km-h',
            'Speed limit 50km-h',
            'Speed limit 60km-h',
            'Speed limit 70km-h',
            'Speed limit 80km-h',
            'End of speed limit 60km-h',
            'End of speed limit 70km-h',
            'End of speed limit 80km-h'
        ]
        # Mapping of custom classes to COCO class IDs (based on YOLOv11 COCO classes)
        self.coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        # Create a mapping from COCO class IDs to custom class IDs
        self.coco_to_custom = {}
        for coco_id, coco_name in enumerate(self.coco_classes):
            if coco_name in self.custom_classes:
                self.coco_to_custom[coco_id] = self.custom_classes.index(coco_name)
            elif coco_name == 'traffic light':  # Map to 'other-vehicle'
                self.coco_to_custom[coco_id] = self.custom_classes.index("green light")

        # Generate distinct colors for each custom class
        self.colors = np.random.uniform(0, 255, size=(len(self.custom_classes), 3))

    def process_frame(self, frame, conf_threshold=0.4):
        """Process a single frame and return detections for custom classes."""
        results = self.model(frame, conf=conf_threshold, iou=0.5)
        detections = []
        
        for result in results:
            for box in result.boxes:
                if box.conf.item() >= conf_threshold:
                    cls_id = int(box.cls.item())
                    if cls_id in self.coco_to_custom:  # Only process mapped classes
                        custom_cls_id = self.coco_to_custom[cls_id]
                        conf = float(box.conf.item())
                        xyxy = box.xyxy[0].tolist()  # Get bounding box coordinates
                        
                        # Convert to integers and clip to frame dimensions
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        detections.append({
                            'class_id': custom_cls_id,
                            'class_name': self.custom_classes[custom_cls_id],
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'color': self.colors[custom_cls_id].tolist()
                        })
        
        return detections

    def process_video(self, video_path, output_dir, conf_threshold=0.4, target_fps=15, display=True):
        """
        Process video file, save unannotated frames and YOLO-format annotations for custom classes.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs
            conf_threshold: Confidence threshold for detections
            target_fps: Target frames per second to process
            display: Whether to show real-time detection results with annotations
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_skip = max(1, int(original_fps / target_fps))

        print(f"Video Info: {frame_width}x{frame_height} @ {original_fps:.1f} FPS")
        print(f"Processing at {target_fps} FPS (1 every {frame_skip} frames)")
        print(f"Total frames to process: ~{total_frames//frame_skip}")

        # Process video frames
        frame_count = 0
        processed_count = 0
        pbar = tqdm(total=total_frames//frame_skip)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to achieve target FPS
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Process frame
            detections = self.process_frame(frame, conf_threshold)
            
            # Save unannotated frame
            frame_name = f"frame_{processed_count:06d}.jpg"
            cv2.imwrite(os.path.join(images_dir, frame_name), frame)

            # Save annotations (YOLO format)
            with open(os.path.join(labels_dir, f"frame_{processed_count:06d}.txt"), 'w') as f:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    x_center = ((x1 + x2) / 2) / frame_width
                    y_center = ((y1 + y2) / 2) / frame_height
                    width = (x2 - x1) / frame_width
                    height = (y2 - y1) / frame_height
                    f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Create annotated frame for display only
            if display:
                annotated_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    color = det['color']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label text
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), color, -1)
                    
                    # Put text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Custom YOLOv11 Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            processed_count += 1
            frame_count += 1
            pbar.update(1)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pbar.close()
        print(f"\nProcessing complete. Saved {processed_count} frames to {output_dir}")

if __name__ == "__main__":
    processor = CustomYOLOv11VideoProcessor(model_path='yolo11x.engine')
    
    # Configuration
    video_path = 'video/test_video.mp4'  # Replace with your video path
    output_directory = 'dataset'  # Output will be saved here
    
    # Process video
    processor.process_video(
        video_path=video_path,
        output_dir=output_directory,
        conf_threshold=0.5,
        target_fps=3,
        display=True
    )