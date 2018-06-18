import math
import numpy as np

from image_processing import visualizer
from image_processing import draw

class GlimpseGenerator():
    def __init__(self):
        pass

    def get_glimpse(self, image, cx, cy, sx, sy, sdx, sdy):
        '''
        Returns the pixels of the glimpse from the image
        cx, cy - position of the center of the glimpse
        sx, sy - arrays of relative position of the kernels in the glimpse
        sdx, sdy - standard deviations of the kernels
        '''
        # Assert that we have a square image
        assert(int(math.sqrt(len(image))) == math.sqrt(int(len(image))))
        # Assert that the length of the xs and ys match
        assert(len(sx) == len(sy))
        pixels = np.zeros(len(sx))
        for i in range(len(sx)):
            pixels[i] = self.get_pixel(image, cx, cy, sx[i], sy[i], sdx[i], sdy[i])
        return pixels

    # TODO - optimize reshape and Gaussian?
    def get_pixel(self, image, cx, cy, sx, sy, sdx, sdy):
        size = int(math.sqrt(len(image)))
        image = np.reshape(image, newshape=(size, size))
        rx = self.get_gaussian_pixel(cx, sx, sdx)
        ry = self.get_gaussian_pixel(cy, sy, sdy)
        # FIXME - take care of the edges of the image - out of bounds error?
        return image[rx][ry]

    # In one of the dimensions x/y
    # c - center of the glimpse
    # ck - center of the kernel
    # sd - standard deviation of the kernel
    def get_gaussian_pixel(self, c, ck, sd):
        return int(np.round(np.random.normal(c - ck, sd)))

# A bit of testing
if __name__ == '__main__':
    import mnist_loader
    x_train, t_train, x_test, t_test = mnist_loader.load("../data/mnist/mnist.pkl")
    glimpser = GlimpseGenerator()
    img = x_train[0]
    glimpse = glimpser.get_glimpse(img, 24, 5, [0], [0], [0], [0])
    print(glimpse.shape)
    print(glimpse)
    size = int(math.sqrt(len(img)))
    visualizer.show_image(np.reshape(img, (size, size)))

    # test one iteration
    draw.draw_circle(np.reshape(img, (size, size)), 10, 10, 2)
    # test multiple iterations
    r = [0, 1, 2, 3, 4]
    for img in x_train[r]:
        draw.draw_circle(np.reshape(img, (size, size)), 10, 10, 2)
