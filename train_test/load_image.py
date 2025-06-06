import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# reading in an image
image = mpimg.imread('lane.png')
# printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
plt.show()

