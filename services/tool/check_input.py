import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg') 
def checkLine(image):
    lane_image = np.copy(image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    candy = cv2.Canny(blur,50,150)
    plt.imshow(candy)
    plt.show()


if __name__ == "__main__":
    # image = cv2.imread("frame_0021.jpg")
    # image = cv2.imread("frame_0021.jpg")
    image = cv2.imread("screen.png")
    checkLine(image)