import numpy as np
import cv2 as cv


def draw_circle(img, xs, ys, radius):
    '''
    :param img: image
    :param xs: pixel position
    :param ys: pixel position
    :param radius:  size of radius
    :return:
    '''
    im = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    im = cv.circle(im, (xs, ys), radius, (0, 0, 255), 1)

    #Display using matplotlib
    #plt.imshow(im)
    #plt.show()

    imS = cv.resize(im, (250, 250))  # Resize window
    cv.imshow('Image', imS)
    cv.waitKey(0)


# Testing
#img = np.zeros((512, 512, 3), np.uint8)
#img = cv.imread('dog.jpeg')
#draw_circle(img, 50, 50, 50)


def create_lattice(x, y, stride, length):
    '''
    :param x:
    :param y:
    :param stride:
    :param length:
    :return: list of indices
    '''

    indices = []
    range_min_to_max = stride * length
    start_y = y - range_min_to_max / 2
    for i in range(length):
        start_x = x - range_min_to_max / 2
        for j in range(length):
            indices.append((start_x, start_y))
            start_x += stride
        start_y += stride

    return indices


# Testing
indices = create_lattice(10, 10, 1, 12)
print(indices)