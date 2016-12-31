import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
import scipy
img = data.astronaut()

x = 225
y = 109
radius = 85
img1 = img.copy()
circle = cv2.circle(img1, (int(x), int(y)), radius, (255, 0, 0),2)
img = rgb2gray(img1)
cv2.imshow("circle", img1)
snake = active_contour(gaussian(img, 3), img1, alpha=0.015, beta=10, gamma=0.001)

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111)
# plt.gray()

# ax.imshow(img)
# ax.plot(init[:, 0], init[:, 1], lw=3)
# ax.plot(snake[:, 0], snake[:, 1], lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])


plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Trial 1'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(img1, cmap='gray')
plt.title('Trial 1'), plt.xticks([]), plt.yticks([])


plt.show()