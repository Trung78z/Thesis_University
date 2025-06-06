import cv2
from matplotlib.pylab import f
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self):
        self.config = {
            'canny_low': 50,
            'canny_high': 150,
            'blur_kernel': (5, 5),
            'roi_bottom_width': 0.90,
            'roi_top_width': 0.35,
            'roi_height': 0.65,
        }

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        bottom_width = self.config['roi_bottom_width'] * width
        top_width = self.config['roi_top_width'] * width
        roi_height = self.config['roi_height'] * height
        # print(f"Image dimensions: {height}x{width}, Bottom width: {bottom_width}, Top width: {top_width}, ROI height: {roi_height}")
        # points = np.array([[
        #     ((width - bottom_width) // 2, height),
        #     ((width - top_width) // 2, int(roi_height)),
        #     ((width + top_width) // 2, int(roi_height)),
        #     ((width + bottom_width) // 2, height),
        # ]], dtype=np.int32)

        points = np.array([[
        (130, height),         # bottom-left
        (580, 460),            # top-left
        (680, 460),            # top-right
        (width - 130, height)  # bottom-right
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

# --- Example usage read video---

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ld = LaneDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        # Resize frame to 1280x720
        frame = cv2.resize(frame, (1280, 720))
        
        # Process frame
        gray, blurred, edges, roi_edges = ld.detect_edges(frame)

        # Display the results
        # cv2.imshow('Original Frame', frame)
        # cv2.imshow('Gray', gray)
        cv2.imshow('Blurred', blurred)
        cv2.imshow('Edges', edges)
        cv2.imshow('ROI Edges', roi_edges)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = '../tool/test_video.mp4'  # Replace with your video file path
    read_video(video_path)