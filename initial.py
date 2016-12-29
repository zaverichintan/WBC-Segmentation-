import numpy as np
import matplotlib.pyplot as plt
import cv2

import glob
# read image
path = 'data/Test_Data/*.jpg'

# files = glob.glob(path)
# for name in files:
    # print name
name = 'data/Train_Data/train-25.jpg'
img = cv2.imread(name,-2)

    # convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold = 65
ret,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)
    #
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    # 	cv2.CHAIN_APPROX_SIMPLE)[-2]
    #
    # print("[INFO] {} unique contours found".format(len(cnts)))
    #
    # for (i, c) in enumerate(cnts):
    # 	# draw the contour
    # 	((x, y), _) = cv2.minEnclosingCircle(c)
    # 	cv2.putText(img, "#{}".format(i + 1), (int(x) - 10, int(y)),
    # 		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # 	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

    # saving the image
    # filename = name.split("/")[2]
    # filename_wo_extention = filename.split(".")[0]
    # filename_wo_extention_w_mask = filename_wo_extention + '-mask.jpg'
    # complete = 'data/submission/'+ filename_wo_extention_w_mask
    # print complete


plt.subplot(111), plt.imshow(thresh, cmap='gray')
plt.title('Trial 1'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.imwrite('thresh.jpg', thresh)
