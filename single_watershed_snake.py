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

# name = 'data/Train_Data/train-28.jpg'
# name = 'data/Train_Data/54A84627F362.jpg'
# name = 'data/mama07ORI.bmp'
name = 'trial.jpg'

img_in = cv2.imread(name,0)
# gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
gray = img_in.copy()
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


img_mask_contour = img_in.copy()
img_mask = img_in.copy()

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
    img_mask_inv = np.zeros(gray.shape, dtype="uint8")

    # Small areas are removed
    if(cv2.contourArea(c)>100):
        ((x, y), r) = cv2.minEnclosingCircle(c)
        r = r + 25
        cv2.circle(img_mask, (int(x), int(y)), int(r), (255, 0, 0), cv2.cv.CV_FILLED, 8, 0)

for i in range(0, len(img_mask)):
    for j in range(0, len(img_mask[0])):
        if (img_mask[i][j] == 255):
            img_mask_inv[i, j] = 255

masked_image = img_in.copy()
for i in range(0, len(img_mask)):
    for j in range(0, len(img_mask[0])):
        if (img_mask[i][j] != 255):
            masked_image[i, j] = 255

plt.subplot(221), plt.imshow(img_in, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(img_mask, cmap='gray')
plt.title('Trial 2'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(img_mask_inv, cmap='gray')
plt.title('mask image'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(masked_image, cmap='gray')
plt.title('Final image'), plt.xticks([]), plt.yticks([])

plt.show()

# g(I)
gI = morphsnakes.gborders(masked_image, alpha=500, sigma=5.88)


cv2.imshow("Mgac", gI)
cv2.waitKey(0)

((x, y), r) = cv2.minEnclosingCircle(c)
r = r - 1
# Morphological GAC. Initialization of the level-set.
mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
mgac.levelset = circle_levelset(masked_image.shape, (x, y), r)

# Visual evolution.
# for i in [1,2,3,4,5]:
#     print i
i = 15
ppl.figure()
morphsnakes.evolve_visual(mgac, num_iters=i, background=masked_image)
ppl.show()
