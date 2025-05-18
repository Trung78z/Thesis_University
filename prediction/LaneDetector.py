import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self, config=None):
        # Default configuration parameters
        self.config = {
            'canny_thresholds': (50, 150),
            'blur_kernel': (5, 5),
            'hough_rho': 2,
            'hough_theta': np.pi/180,
            'hough_threshold': 100,
            'min_line_length': 40,
            'max_line_gap': 5,
            'roi_vertices': None,  # Will be set based on image size
            'line_color': (255, 150, 0),
            'line_thickness': 10,
            'y_bottom_ratio': 0.95,
            'y_top_ratio': 0.45,
            'slope_threshold': 0.5
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
            
        # Placeholders for previous frame data
        self.prev_left = None
        self.prev_right = None
        self.smoothing_factor = 0.2

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = int(image.shape[0] * self.config['y_bottom_ratio'])
        y2 = int(image.shape[0] * self.config['y_top_ratio'])
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []

        if lines is None:
            # Return previous lines if available
            if self.prev_left is not None and self.prev_right is not None:
                return np.array([self.prev_left, self.prev_right])
            return None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters
            
            # Filter lines based on slope threshold
            if abs(slope) < self.config['slope_threshold']:
                continue
                
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        averaged_lines = []

        # Process left lane
        if left_fit:
            left_fit_avg = np.average(left_fit, axis=0)
            left_line = self.make_coordinates(image, left_fit_avg)
            
            # Apply smoothing with previous frame
            if self.prev_left is not None:
                left_line = ((1 - self.smoothing_factor) * left_line + 
                            self.smoothing_factor * self.prev_left)
            
            averaged_lines.append(left_line)
            self.prev_left = left_line
        elif self.prev_left is not None:
            averaged_lines.append(self.prev_left)

        # Process right lane
        if right_fit:
            right_fit_avg = np.average(right_fit, axis=0)
            right_line = self.make_coordinates(image, right_fit_avg)
            
            # Apply smoothing with previous frame
            if self.prev_right is not None:
                right_line = ((1 - self.smoothing_factor) * right_line + 
                             self.smoothing_factor * self.prev_right)
            
            averaged_lines.append(right_line)
            self.prev_right = right_line
        elif self.prev_right is not None:
            averaged_lines.append(self.prev_right)

        return np.array(averaged_lines) if averaged_lines else None

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, self.config['blur_kernel'], 0)
        return cv2.Canny(blur, *self.config['canny_thresholds'])

    def region_of_interest(self, image):
        height = image.shape[0]

        self.config.setdefault('x1', 200)
        self.config.setdefault('x2', 1100)
        self.config.setdefault('x3', 470)
        self.config.setdefault('x4', 730)
        self.config.setdefault('y', 400)

        self.config['roi_vertices'] = np.array([[
            (self.config['x1'], height),
            (self.config['x2'], height),
            (self.config['x4'], self.config['y']),
            (self.config['x3'], self.config['y'])
        ]], dtype=np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, self.config['roi_vertices'], 255)
        return cv2.bitwise_and(image, mask)

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.astype(int)
                cv2.line(line_image, (x1, y1), (x2, y2), 
                         self.config['line_color'], 
                         self.config['line_thickness'])
        return line_image

    def process_image(self, image):
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        lane_image = np.copy(image)
        canny_image = self.canny(lane_image)
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
        
        averaged_lines = self.average_slope_intercept(lane_image, lines)
        line_image = self.display_lines(lane_image, averaged_lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        
        return combo_image, averaged_lines

    def visualize(self, image):
        result, _ = self.process_image(image)
        plt.imshow(result)
        plt.show()