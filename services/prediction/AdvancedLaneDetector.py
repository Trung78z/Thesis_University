import cv2
import numpy as np
from scipy import stats
from ultralytics import YOLO
from collections import deque

class AdvancedLaneDetector:
    def __init__(self, yolo_model_path="yolov8n.pt", config=None):
        # Initialize YOLOv8 model
        try:
            self.yolo_model = YOLO(yolo_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
        
        self.target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # car (2) and traffic light (9)
        
        # Optimized configuration for lane detection
        self.config = {
            'canny_low': 50,              # Lower edge sensitivity
            'canny_high': 150,            # Higher edge strength
            'blur_kernel': (5, 5),        # Noise reduction
            'hough_rho': 1,               # Fine distance resolution
            'hough_theta': np.pi/180,     # Precise angle resolution
            'hough_threshold': 40,        # Reduced for more line detection
            'min_line_length': 30,        # Filters short noise lines
            'max_line_gap': 25,           # Connects slightly broken lines
            'line_color': (0, 255, 255),  # Yellow lines
            'line_thickness': 4,          # Visible line thickness
            'fill_color': (0, 255, 0),    # Green lane fill
            'fill_alpha': 0.3,            # Subtle transparency
            'slope_threshold': 0.4,       # Stricter slope filtering
            'confidence_window': 5,       # Smooth over 5 frames
            'roi_bottom_width': 0.85,     # Wider base for highway lanes
            'roi_top_width': 0.35,        # Narrower top for focus
            'roi_height': 0.62,           # Captures more road area
            'min_detection_confidence': 0.5,  # Stricter confidence
            'max_fps': 30,                # Real-time target
        }
        
        if config:
            self.config.update(config)
            
        # State management
        self.prev_left = None
        self.prev_right = None
        self.left_history = deque(maxlen=self.config['confidence_window'])
        self.right_history = deque(maxlen=self.config['confidence_window'])
        self.frame_counter = 0
        self.fps = 0
        self.processing_time = 0

    def _robust_line_fit(self, points):
        """Safe line fitting with comprehensive error handling"""
        if points is None or len(points) < 2:
            return None
            
        try:
            x = points[:, 0]
            y = points[:, 1]
            
            if np.std(x) < 1e-6:  # Near-vertical lines
                return (float('inf'), np.mean(x)), 1.0
                
            if np.std(y) < 1e-6:  # Near-horizontal lines
                return (0, np.mean(y)), 1.0
                
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            confidence = r_value**2
            if confidence < self.config['min_detection_confidence']:
                return None
                
            return (slope, intercept), confidence
            
        except Exception:
            return None

    def _filter_lines(self, lines):
        """Filter and classify valid lane lines"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
            
        for line in lines:
            try:
                x1, y1, x2, y2 = line.reshape(4)
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length < self.config['min_line_length']:
                    continue
                    
                if x1 == x2:  # Skip vertical lines
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                if abs(slope) < self.config['slope_threshold']:
                    continue
                    
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))
            except Exception:
                continue
                
        return left_lines, right_lines

    def _calculate_lane(self, lines, history, prev_lane):
        """Calculate lane position with temporal smoothing"""
        if not lines:
            if prev_lane is not None:
                return prev_lane, 0.3
            return None, 0.0
            
        points = []
        for slope, intercept in lines:
            y1 = int(self.height * self.config['roi_height'])
            y2 = self.height
            try:
                x1 = int((y1 - intercept)/slope) if abs(slope) > 1e-6 else 0
                x2 = int((y2 - intercept)/slope) if abs(slope) > 1e-6 else 0
                points.extend([(x1, y1), (x2, y2)])
            except Exception:
                continue
                
        if len(points) < 2:
            if prev_lane is not None:
                return prev_lane, 0.2
            return None, 0.0
            
        points = np.array(points)
        result = self._robust_line_fit(points)
        
        if result is None:
            if prev_lane is not None:
                return prev_lane, 0.2
            return None, 0.0
            
        (slope, intercept), confidence = result
        
        history.append((slope, intercept))
        if len(history) > 1:
            slope = np.mean([h[0] for h in history])
            intercept = np.mean([h[1] for h in history])
            confidence = min(1.0, confidence * (1 + len(history)/10))
            
        return (slope, intercept), confidence

    def _get_roi_mask(self):
        """Generate dynamic region of interest mask"""
        bottom_left = (int(self.width*(1-self.config['roi_bottom_width'])/2), self.height)
        bottom_right = (int(self.width*(1+self.config['roi_bottom_width'])/2), self.height)
        top_right = (int(self.width*(1+self.config['roi_top_width'])/2), int(self.height*self.config['roi_height']))
        top_left = (int(self.width*(1-self.config['roi_top_width'])/2), int(self.height*self.config['roi_height']))
        
        vertices = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.int32)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def _detect_lanes(self, frame):
        """Core lane detection pipeline"""
        try:
            self.height, self.width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, self.config['blur_kernel'], 0)
            edges = cv2.Canny(blur, self.config['canny_low'], self.config['canny_high'])
            roi_mask = self._get_roi_mask()
            masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
            
            lines = cv2.HoughLinesP(
                masked_edges,
                self.config['hough_rho'],
                self.config['hough_theta'],
                self.config['hough_threshold'],
                minLineLength=self.config['min_line_length'],
                maxLineGap=self.config['max_line_gap']
            )
            
            left_lines, right_lines = self._filter_lines(lines)
            left_lane, left_conf = self._calculate_lane(left_lines, self.left_history, self.prev_left)
            right_lane, right_conf = self._calculate_lane(right_lines, self.right_history, self.prev_right)
            
            if left_lane:
                self.prev_left = left_lane
            if right_lane:
                self.prev_right = right_lane
                
            return left_lane, right_lane, min(left_conf, right_conf)
            
        except Exception as e:
            print(f"Lane detection error: {str(e)}")
            return self.prev_left, self.prev_right, 0.0

    def _draw_lanes(self, frame, left_lane, right_lane, confidence):
        """Visualize detected lanes"""
        overlay = frame.copy()
        
        if left_lane is not None and right_lane is not None:
            try:
                slope_left, intercept_left = left_lane
                slope_right, intercept_right = right_lane
                
                y1 = int(self.height * self.config['roi_height'])
                y2 = self.height
                
                x1_left = max(0, min(self.width, int((y1 - intercept_left)/slope_left)))
                x2_left = max(0, min(self.width, int((y2 - intercept_left)/slope_left)))
                x1_right = max(0, min(self.width, int((y1 - intercept_right)/slope_right)))
                x2_right = max(0, min(self.width, int((y2 - intercept_right)/slope_right)))
                
                pts = np.array([
                    [x1_left, y1], [x2_left, y2],
                    [x2_right, y2], [x1_right, y1]
                ], np.int32)
                
                fill_color = (
                    int(self.config['fill_color'][0] * confidence),
                    int(self.config['fill_color'][1] * confidence),
                    int(self.config['fill_color'][2] * confidence)
                )
                cv2.fillPoly(overlay, [pts], fill_color)
                
                cv2.line(overlay, (x1_left, y1), (x2_left, y2), 
                        self.config['line_color'], self.config['line_thickness'])
                cv2.line(overlay, (x1_right, y1), (x2_right, y2), 
                        self.config['line_color'], self.config['line_thickness'])
            except Exception:
                pass
                
        cv2.addWeighted(overlay, self.config['fill_alpha'], frame, 1 - self.config['fill_alpha'], 0, frame)
        return frame

    def process_frame(self, frame):
        """Process a single frame"""
        try:
            start_time = cv2.getTickCount()
            
            if frame is None or frame.size == 0:
                print("Warning: Empty frame received")
                return None
                
            frame = cv2.resize(frame, (1280, 720))  # Resize for consistency
            self.height, self.width = frame.shape[:2]
            
            # YOLO detection
            try:
                yolo_results = self.yolo_model(frame, classes=self.target_classes, verbose=False)
            except Exception as e:
                print(f"YOLO detection error: {str(e)}")
                yolo_results = []
            
            # Lane detection
            left_lane, right_lane, lane_confidence = self._detect_lanes(frame)
            frame = self._draw_lanes(frame, left_lane, right_lane, lane_confidence)
            
            # Draw YOLO detections
            for result in yolo_results:
                try:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        cls_id = int(box.cls)
                        
                        if conf > 0.5:
                            color = (0, 255, 0) if cls_id == 2 else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{self.yolo_model.names[cls_id]} {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception:
                    continue
            
            # FPS calculation
            self.processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return None