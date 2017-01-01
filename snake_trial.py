import morphsnakes

import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl
from matplotlib import pyplot as plt
import cv2


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
    background : array-like, optional
        If given, background will be shown behind the contours instead of
        msnake.data.
    """
    from matplotlib import pyplot as ppl

    if levelset is not None:
        msnake.levelset = levelset

    # Prepare the visual environment.
    fig = ppl.gcf()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    if background is None:
        ax1.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax1.imshow(background, cmap=ppl.cm.gray)
    ax1.contour(msnake.levelset, [0.5], colors='r')

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(msnake.levelset)
    ppl.pause(0.001)

    # Iterate.
    for i in range(num_iters):
        # Evolve.
        msnake.step()

        # Update figure.
        # del ax1.collections[0]
        ax1.contour(msnake.levelset, [0.5], colors='r')
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()
        # we get the image here
        cv2.imshow("snake", msnake.levelset)
        #  ppl.pause(0.001)

    # Return the last levelset.
    return msnake.levelset


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u


def test_nodule():
    # Load the image.
    # img = imread("data/mama07ORI.bmp")[..., 0] / 255.0
    name = 'data/Train_Data/train-1.jpg'
    img = cv2.imread(name, 0)
    # g(I)
    gI = morphsnakes.gborders(img, alpha=100, sigma=1.48)
    cv2.imshow("GI", gI)
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)

    final_image = np.zeros(img.shape, dtype="uint8")


    mgac.levelset = circle_levelset(img.shape, (64, 60), 20)
    # Visual evolution.
    ppl.figure()
    snake_image = evolve_visual(mgac, num_iters=40, background=img)
    ppl.show()

    for i in range(0, snake_image.shape[0]):
        for j in range(0, snake_image.shape[1]):
            if (snake_image[i][j] == 1):
                final_image[i][j] = 255

    mgac.levelset = circle_levelset(img.shape, (34, 60), 20)
    # Visual evolution.
    ppl.figure()
    snake_image = evolve_visual(mgac, num_iters=40, background=img)
    ppl.show()

    for i in range(0, snake_image.shape[0]):
        for j in range(0, snake_image.shape[1]):
            if (snake_image[i][j] == 1):
                final_image[i][j] = 255

    plt.subplot(111), plt.imshow(final_image)
    plt.title('Final image'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    print("""""")
    test_nodule()