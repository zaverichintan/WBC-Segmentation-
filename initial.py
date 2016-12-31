import numpy as np
import matplotlib.pyplot as plt
import cv2

import glob
# read image
path = 'data/Test_Data/*.jpg'

# files = glob.glob(path)
# for name in files:
    # print name
name = 'data/Train_Data/train-1.jpg'
img = cv2.imread(name,-2)

    # convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold = 65
ret,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)


plt.subplot(111), plt.imshow(thresh, cmap='gray')
plt.title('Trial 1'), plt.xticks([]), plt.yticks([])

plt.show()
# cv2.imwrite('thresh.jpg', thresh)
