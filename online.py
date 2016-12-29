import morphsnakes

import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl
import cv2

# def rgb2gray(img):
#     """Convert a RGB image to gray scale."""
#     return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
#

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    print u 
    return u


# def test_nodule():
    # Load the image.
name = 'data/Train_Data/train-25.jpg'

img = imread(name)[..., 0] / 255.0

# g(I)
gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)
cv2.imshow('imag',gI)

# Morphological GAC. Initialization of the level-set.
mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
mgac.levelset = circle_levelset(img.shape, (69, 74), 20)

# Visual evolution.
ppl.figure()
morphsnakes.evolve_visual(mgac, num_iters=45, background=img)

if __name__ == '__main__':
    # test_nodule()
    ppl.show()
