import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

# read image
# name = 'data/Train_Data/54A84627F362.jpg'
name = 'data/Train_Data/train-25.jpg'
tobeobtained_name = 'data/Train_Data/train-25-mask.jpg'
img_in = cv2.imread(name,-1)
tobeobtained = cv2.imread(tobeobtained_name,-1)
# convert image to grayscale
gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

threshold = 65
ret,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
                          labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)


masked_image = img_in.copy()
# loop over the unique labels returned by the Watershed algorithm
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue

    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)[-2]
    c = max(cnts, key=cv2.contourArea)
    if(cv2.contourArea(c)>100):
        cv2.drawContours(masked_image, [c], 0, (0, 255, 0), 1)
        area = cv2.contourArea(c)

    final_image = np.zeros(gray.shape, dtype="uint8")


plt.subplot(221), plt.imshow(img_in, cmap='gray')
plt.title('Trial 1'), plt.xticks([]), plt.yticks([])

# plt.subplot(222), plt.imshow(, cmap='gray')
# plt.title('Trial 2'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(final_image, cmap='gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])


plt.subplot(224), plt.imshow(masked_image ,cmap='gray')
plt.title('To be obtatined'), plt.xticks([]), plt.yticks([])


plt.show()