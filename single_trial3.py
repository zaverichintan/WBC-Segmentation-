import morphsnakes
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl
from matplotlib import pyplot as plt

import cv2

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    print u
    return u


# def test_nodule():
    # Load the image.
# name = 'data/Train_Data/train-25.jpg'
name = 'data/Train_Data/train-1.jpg'
tobeobtained_name = 'data/Train_Data/train-1-mask.jpg'

# name = 'data/Train_Data/54A84627F362.jpg'
# img = imread(name)[..., 0] / 255.0
img = cv2.imread(name, -1)
tobeobtained = cv2.imread(tobeobtained_name,-1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


# img_mask_contour = img_in.copy()
# img_mask = img_in.copy()

# loop over the unique labels returned by the Watershed algorithm
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue

    # allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)[-2]
    c = max(cnts, key=cv2.contourArea)
    final_image = np.zeros(gray.shape, dtype="uint8")

    # Small areas are removed
    if(cv2.contourArea(c)>100):
        ((x, y), r) = cv2.minEnclosingCircle(c)
        # r = r + 5

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Trial 1'), plt.xticks([]), plt.yticks([])

# plt.subplot(122), plt.imshow(img_mask, cmap='gray')
# plt.title('Trial 2'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(223), plt.imshow(masked_image, cmap='gray')
# plt.title('mask'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(224), plt.imshow(tobeobtained, cmap='gray')
# plt.title('To be obtatined'), plt.xticks([]), plt.yticks([])
plt.show()

# g(I)
gI = morphsnakes.gborders(img, alpha=1000, sigma=5.88)

# plt.subplot(111), plt.imshow(gI, cmap='gray')
# plt.title('To be obtatined'), plt.xticks([]), plt.yticks([])
# plt.show()

# ((x, y), r) = cv2.minEnclosingCircle(c)
# r = r - 1

mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
print img.shape

mgac.levelset = circle_levelset(img.shape, (x, y), r)

# Visual evolution.
ppl.figure()
morphsnakes.evolve_visual(mgac, num_iters=55, background=img)
ppl.show()

plt.show()