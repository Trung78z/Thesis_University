import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LaneDetector:
    def __init__(self, config=None):
        # Default configuration parameters
        self.config = {
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'blur_kernel': (5, 5),
            'hough_rho': 1,
            'hough_theta': np.pi/180,
            'hough_threshold': 80,
            'min_line_length': 30,
            'max_line_gap': 10,
            'line_color': (0, 165, 255),  # Orange for visibility
            'line_thickness': 8,
            'y_bottom_ratio': 0.95,
            'y_top_ratio': 0.55,
            'slope_threshold': 0.4,
            'confidence_threshold': 0.7,  # For adaptive smoothing
            'fill_alpha': 0.3,  # Transparency for lane fill
        }
        
        if config:
            self.config.update(config)
            
        self.prev_left = None
        self.prev_right = None
        self.smoothing_factor = 0.3
        self.frame_count = 0

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        height = image.shape[0]
        y1 = int(height * self.config['y_bottom_ratio'])
        y2 = int(height * self.config['y_top_ratio'])
        x1 = int((y1 - intercept) / slope) if slope != 0 else 0
        x2 = int((y2 - intercept) / slope) if slope != 0 else 0
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []

        if lines is None:
            if self.prev_left is not None and self.prev_right is not None:
                return np.array([self.prev_left, self.prev_right])
            return None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 == x2:  # Avoid division by zero
                continue
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters
            
            # Filter by slope and length
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if abs(slope) < self.config['slope_threshold'] or line_length < self.config['min_line_length']:
                continue
                
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        averaged_lines = []
        confidence = 1.0

        def robust_average(fit_list):
            if not fit_list:
                return None, 0.0
            fit_array = np.array(fit_list)
            # Use RANSAC to reject outliers
            if len(fit_array) > 5:
                slopes = fit_array[:, 0]
                inliers = stats.zscore(slopes) < 2  # Remove outliers > 2 std deviations
                fit_array = fit_array[inliers]
            if len(fit_array) == 0:
                return None, 0.0
            avg = np.average(fit_array, axis=0)
            conf = len(fit_array) / max(1, len(lines))  # Confidence based on number of valid lines
            return avg, conf

        # Process left lane
        left_fit_avg, left_conf = robust_average(left_fit)
        if left_fit_avg is not None:
            left_line = self.make_coordinates(image, left_fit_avg)
            if self.prev_left is not None:
                adaptive_smooth = self.smoothing_factor * (1 - left_conf)
                left_line = (1 - adaptive_smooth) * left_line + adaptive_smooth * self.prev_left
            averaged_lines.append(left_line)
            self.prev_left = left_line
        elif self.prev_left is not None:
            averaged_lines.append(self.prev_left)
            confidence = min(confidence, 0.5)

        # Process right lane
        right_fit_avg, right_conf = robust_average(right_fit)
        if right_fit_avg is not None:
            right_line = self.make_coordinates(image, right_fit_avg)
            if self.prev_right is not None:
                adaptive_smooth = self.smoothing_factor * (1 - right_conf)
                right_line = (1 - adaptive_smooth) * right_line + adaptive_smooth * self.prev_right
            averaged_lines.append(right_line)
            self.prev_right = right_line
        elif self.prev_right is not None:
            averaged_lines.append(self.prev_right)
            confidence = min(confidence, 0.5)

        self.frame_count += 1
        return np.array(averaged_lines) if averaged_lines else None, min(left_conf, right_conf)

    def adaptive_canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, self.config['blur_kernel'], 0)
        # Adaptive thresholding based on image median intensity
        median_intensity = np.median(gray)
        low_threshold = max(20, min(100, int(median_intensity * 0.33)))
        high_threshold = min(255, low_threshold * 3)
        return cv2.Canny(blur, low_threshold, high_threshold)

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        # Dynamic ROI based on image dimensions
        vertices = np.array([[
            (width * 0.1, height),  # Bottom-left
            (width * 0.9, height),  # Bottom-right
            (width * 0.6, height * 0.6),  # Top-right
            (width * 0.4, height * 0.6)   # Top-left
        ]], dtype=np.int32)
        
        mask = np.zeros_like(image)
        channel_count = image.shape[2] if len(image.shape) == 3 else 1
        fill_color = 255 if channel_count == 1 else (255, 255, 255)
        cv2.fillPoly(mask, vertices, fill_color)
        return cv2.bitwise_and(image, mask)

    def display_lines(self, image, lines, confidence):
        line_image = np.zeros_like(image)
        if lines is not None:
            # Draw filled polygon between lanes for visualization
            if len(lines) == 2:
                points = np.array([
                    [lines[0][0], lines[0][1]],  # Left bottom
                    [lines[0][2], lines[0][3]],  # Left top
                    [lines[1][2], lines[1][3]],  # Right top
                    [lines[1][0], lines[1][1]]   # Right bottom
                ], dtype=np.int32)
                cv2.fillPoly(line_image, [points], 
                            (0, int(255 * confidence), 0),  # Green fill with confidence-based intensity
                            lineType=cv2.LINE_AA)
            
            # Draw lane lines
            for line in lines:
                x1, y1, x2, y2 = line.astype(int)
                cv2.line(line_image, (x1, y1), (x2, y2), 
                        self.config['line_color'], 
                        self.config['line_thickness'],
                        lineType=cv2.LINE_AA)
        return line_image

    def preprocess_image(self, image):
        # Convert to HSV and use V channel for better robustness to lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, _, v = cv2.split(hsv)
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(v)

    def process_image(self, image):
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        lane_image = np.copy(image)
        
        # Preprocess with HSV and CLAHE
        preprocessed = self.preprocess_image(lane_image)
        canny_image = self.adaptive_canny(lane_image)
        cropped_image = self.region_of_interest(canny_image)
        
        lines = cv2.HoughLinesP(
            cropped_image,
            self.config['hough_rho'],
            self.config['hough_theta'],
            self.config['hough_threshold'],
            np.array([]),
            minLineLength=self.config['min_line_length'],
            maxLineGap=self.config['max_line_gap']
        )
        
        averaged_lines, confidence = self.average_slope_intercept(lane_image, lines)
        line_image = self.display_lines(lane_image, averaged_lines, confidence)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        
        return combo_image, averaged_lines

    def visualize(self, image):
        result, _ = self.process_image(image)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Lane Detection Result')
        plt.show()