import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

import glob
# read image
# path = 'data/Test_Data/*.jpg'
path = 'data/Train_Data/*.jpg'

files = glob.glob(path)
for name in files:
    # print name
    img = cv2.imread(name, -1)

    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold = 65
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

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
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(c)

        # initializing final image
        final_image = np.zeros(gray.shape, dtype="uint8")

        # removing small areas
        if (area > 100):
            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            radius = int(r) + 5
            print radius
            # cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), 2)
            cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), cv2.cv.CV_FILLED, 8, 0)

    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if (img[i][j][0] == 255):
                final_image[i, j] = 255

    # saving the image
    filename = name.split("/")[2]
    filename_wo_extention = filename.split(".")[0]
    filename_wo_extention_w_mask = filename_wo_extention + '-mask.jpg'
    complete = 'data/training_mask/' + filename_wo_extention_w_mask
    print complete
    cv2.imwrite(complete, final_image)