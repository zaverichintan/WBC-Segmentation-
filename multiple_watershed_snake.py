import cv2
import morphsnakes
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
from scipy.misc import imread
from skimage import feature
from skimage import img_as_ubyte
import glob

def evolve_visual(msnake, levelset=None, num_iters=20, background=None):
    """
    Visual evolution of a morphological snake.

    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    """

    if levelset is not None:
        msnake.levelset = levelset

    # Iterate.
    for i in range(num_iters):
        # Evolve.
        msnake.step()
    # Return the last levelset.
    return msnake.levelset

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u



path = 'data/Test_Data/*.jpg'

files = glob.glob(path)
for name in files:

    img_in = cv2.imread(name, 0)
    gray = img_in.copy()
    threshold = 65
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    # compute the exact Euclidean distance from every binary pixel to the nearest zero pixel, then find peaks in this distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    # perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    # Declaring variables
    img_mask_contour = img_in.copy()
    img_mask = img_in.copy()
    positions = []
    final_image = np.zeros(img_in.shape, dtype="uint8")
    # loop over the unique labels returned by the Watershed algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # allocate memory for the label region and draw it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[-2]
        c = max(cnts, key=cv2.contourArea)
        img_mask_inv = np.zeros(gray.shape, dtype="uint8")

        for c_each in cnts:
            # Small areas are removed
            if (cv2.contourArea(c) > 100 and cv2.contourArea(c) < 5000):
                ((x, y), r) = cv2.minEnclosingCircle(c)
                # print x,y,r
                r = r + 15
                cv2.circle(img_mask, (int(x), int(y)), int(r), (255, 0, 0), cv2.cv.CV_FILLED, 8, 0)
                positions.append([x, y, r])

    # It stores the positions of the contours
    positions = np.array(positions)

    masked_image = img_in.copy()
    for i in range(0, len(img_mask)):
        for j in range(0, len(img_mask[0])):
            if (img_mask[i][j] != 255):
                masked_image[i, j] = 255

    # Apply canny filter to detect edges
    edges = feature.canny(masked_image, sigma=3)
    cv_edges = img_as_ubyte(edges)
    kernel = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(cv_edges, kernel, iterations=1)

    cnts = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    cv2.drawContours(masked_image, cnts, -1, (255, 0, 0), 2)

    # Applying snakes algorithm
    for i in range(0, len(positions)):
        # g(I)
        gI = morphsnakes.gborders(masked_image, alpha=1000, sigma=5.88)
        x = positions[i][1]  # Inversion
        y = positions[i][0]
        r = positions[i][2]
        # Morphological GAC. Initialization of the level-set.
        mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
        mgac.levelset = circle_levelset(masked_image.shape, (x, y), r)
        snake_image = evolve_visual(mgac, num_iters=5, background=masked_image)
        for i in range(0, snake_image.shape[0]):
            for j in range(0, snake_image.shape[1]):
                if (snake_image[i][j] == 1):
                    final_image[i][j] = 255

    # saving the image
    filename = name.split("/")[2]
    filename_wo_extention = filename.split(".")[0]
    filename_wo_extention_w_mask = filename_wo_extention + '-mask.jpg'
    complete = 'data/testing_mask/' + filename_wo_extention_w_mask
    print complete
    cv2.imwrite(complete, final_image)