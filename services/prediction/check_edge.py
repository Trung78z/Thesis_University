import cv2
from matplotlib.pylab import f
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self):
        self.config = {
            'canny_low': 30,
            'canny_high': 100,
            'blur_kernel': (7, 7),
            'roi_bottom_width': 0.90,
            'roi_top_width': 0.35,
            'roi_height': 0.65,
        }

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        bottom_width = self.config['roi_bottom_width'] * width
        top_width = self.config['roi_top_width'] * width
        roi_height = self.config['roi_height'] * height

        points = np.array([[
            ((width - bottom_width) // 2, height),
            ((width - top_width) // 2, int(roi_height)),
            ((width + top_width) // 2, int(roi_height)),
            ((width + bottom_width) // 2, height),
        ]], dtype=np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, points, 255)
        masked = cv2.bitwise_and(image, mask)
        return masked

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config['blur_kernel'], 0)
        edges = cv2.Canny(blurred, self.config['canny_low'], self.config['canny_high'])
        roi_edges = self.region_of_interest(edges)
        return gray, blurred, edges, roi_edges

# --- Example usage ---
frame = cv2.imread("road_sample.png")  # Read demo image
frame = cv2.resize(frame, (1280, 720))  # Resize frame to 1280x720
ld = LaneDetector()
gray, blurred, edges, roi_edges = ld.detect_edges(frame)

# Convert BGR to RGB for matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Create subplots
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(frame_rgb)
plt.title('Original Image')
plt.axis('off')

# Grayscale Image
plt.subplot(2, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Blurred Image
plt.subplot(2, 2, 3)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

# Canny Edges with ROI
plt.subplot(2, 2, 4)
plt.imshow(roi_edges, cmap='gray')
plt.title('Canny Edges with ROI')
plt.axis('off')

plt.tight_layout()
plt.show()