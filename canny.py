import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
from skimage import feature
np.set_printoptions(threshold=np.inf)

name = 'data/Train_Data/train-1.jpg'
# name = 'data/Train_Data/54A84627F362.jpg'
# name = 'data/Train_Data/1467973104.jpg'

img = cv2.imread(name,0)
# Compute the Canny filter for two/ values of sigma
edges = feature.canny(img, sigma=3)
from skimage import img_as_ubyte
cv_edges = img_as_ubyte(edges)
kernel = np.ones((5,5), np.uint8)

img_dilation = cv2.dilate(cv_edges, kernel, iterations=1)

cnts = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2]
cv2.drawContours(img,cnts, -1 , (255,0,0), 2)

cv2.imwrite("trial.jpg", img)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(img_dilation, cmap='gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])


plt.show()





















# old code


# img = cv2.GaussianBlur(img,(3,3),0)
#
# # convolute with proper kernels
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
#
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#
# plt.show()
#
# # display results
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
#
# ax1.imshow(im, cmap=plt.cm.jet)
# ax1.axis('off')
# ax1.set_title('noisy image', fontsize=20)
#
# ax2.imshow(edges1, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)
#
# ax3.imshow(edges2, cmap=plt.cm.gray)
# ax3.axis('off')
# ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)
#
# fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
#                     bottom=0.02, left=0.02, right=0.98)
#
# plt.show()