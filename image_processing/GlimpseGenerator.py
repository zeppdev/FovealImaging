import math
import random

import numpy as np

from image_processing import visualizer
from image_processing import draw
import pickle



class GlimpseGenerator():
    def __init__(self):
        pass

    def get_glimpse(self, image, cx, cy, sx, sy, std):
        '''
        Returns the pixels of the glimpse from the image
        cx, cy - position of the center of the glimpse
        sx, sy - arrays of relative position of the kernels in the glimpse
        sdx, sdy - standard deviations of the kernels
        '''

        # TODO - clipping the input might not be the correct way to approach this...
        cx = np.clip(cx, 0, None)
        cy = np.clip(cy, 0, None)
        std = np.abs(std)

        # Assert that we have a square image
        assert(int(math.sqrt(len(image))) == math.sqrt(int(len(image))))
        # Assert that the length of the xs and ys match
        assert(len(sx) == len(sy))
        pixels = np.zeros(len(sx))
        for i in range(len(sx)):
            pixels[i] = self.get_pixel(image, cx, cy, sx[i], sy[i], std[i], std[i])
        return pixels

    # TODO - optimize reshape and Gaussian?
    def get_pixel(self, image, cx, cy, sx, sy, sdx, sdy):
        size = int(math.sqrt(len(image)))
        image = np.reshape(image, newshape=(size, size))
        rx, ry = self.get_gaussian_pixel([cx - sx, cy - sy], [sdx, sdx])
        # ry = self.get_gaussian_pixel(cy, sy, sdy)
        rx = np.clip(rx, 0, size-1)
        ry = np.clip(ry, 0, size-1)
        # FIXME - take care of the edges of the image - out of bounds error?
        return image[int(rx)][int(ry)]

    # In one of the dimensions x/y
    # c - center of the glimpse
    # ck - center of the kernel
    # sd - standard deviation of the kernel
    def get_gaussian_pixel(self, c, sd):
        return np.round(np.random.normal(c, sd))

# A bit of testing
if __name__ == '__main__':
    from timeit import default_timer as timer
    np.random.seed(42)

    # import mnist_loader
    # x_train, t_train, x_test, t_test = mnist_loader.load("../data/mnist/mnist.pkl")
    glimpser = GlimpseGenerator()
    img = np.random.randint(0, 255, 784)
    xs = np.random.randint(-12, 12, 64)
    ys = np.random.randint(-12, 12, 64)
    stdx = np.random.rand(144,)
    start = timer()
    glimpser = GlimpseGenerator()

    glimpses = []
    for i in range(10000):
        glimpses.append(glimpser.get_glimpse(img, 24, 5, xs, ys, stdx))
    end = timer()
    print("Time taken:",  (end - start))
    pickle.dump(glimpses, open("glimpses", mode="wb"))
    # print(glimpse.shape)
    # print(glimpse)
    # size = int(math.sqrt(len(img)))
    # visualizer.show_image(np.reshape(img, (size, size)))
    #
    # # test one iteration
    # draw.draw_circle(np.reshape(img, (size, size)), 10, 10, 2)
    # # test multiple iterations
    # r = [0, 1, 2, 3, 4]
    # for img in x_train[r]:
    #     draw.draw_circle(np.reshape(img, (size, size)), 10, 10, 2)
