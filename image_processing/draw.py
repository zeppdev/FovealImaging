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
    #img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    im = cv.circle(img, (xs, ys), radius, (0, 0, 255), -1)

    #Display using matplotlib
    #plt.imshow(im)
    #plt.show()

    # Display using opencv
    imS = cv.resize(im, (250, 250))  # Resize window
    cv.imshow('Image', im)


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
            indices.append((int(start_x), int(start_y)))
            start_x += stride
        start_y += stride

    return indices



# Testing

img = np.zeros((512, 512, 3), np.uint8)
overlay_t = cv.imread('../data/five.png', -1)  # -1 loads with transparency

indices = create_lattice(10, 10, 1, 12)

for i in range(len(indices)):
    draw_circle(img, int(indices[i][0]), int(indices[i][1]), 2)
