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

    imS = cv.resize(im, (250, 250))  # Resize window
    cv.imshow('Image', imS)
    cv.waitKey(0)


# Testing
#img = np.zeros((512, 512, 3), np.uint8)
#img = cv.imread('dog.jpeg')
#draw_circle(img, 50, 50, 50)