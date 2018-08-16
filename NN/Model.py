import math

import cv2
import numpy as np

import keras.backend as K
import time
from keras import Sequential, Input
from keras.layers import RNN, Dense, SimpleRNN, Masking, LSTM, TimeDistributed
from image_processing import mnist_loader
from image_processing.GlimpseGenerator import GlimpseGenerator

from matplotlib import pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model():

    def __init__(self, logging=False):
        self.logging = logging

        # Initial standard deviation of the kernel
        self.kernel_init_std = 1
        self.kernel_weights = None
        self.k_size = 8

        self.rnn_timesteps = 4
        self.rnn_layers = [256, 256]
        self.rnn_activation = 'relu'

        self.with_zoom = False
        self.control_output = 3 if self.with_zoom else 2

        self.batch_size = 64
        self.path_to_images = "data/mnist/mnist.pkl"
        self.image_size = 28
        self.nr_of_classes = 10

        # gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = K.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = K.tf.Session(config=config)
        with self.sess.as_default():
            self.glimpse = K.variable(np.zeros((1, 1, self.k_size ** 2)))
            self.init_image_loader()
            self.init_networks()
            self.init_weight_sizes()

    def init_weight_sizes(self):
        # 3 because we need x, y coordinates and std in both directions
        self.kernel_weight_size = 3 * self.k_size ** 2
        self.rnn_weight_size = sum([w.size for w in self.rnn_model.get_weights()])
        self.control_weight_size = sum([w.size for w in self.control_model.get_weights()])
        self.classifier_weight_size = sum([w.size for w in self.classifier_model.get_weights()])
        self.weights_size = self.kernel_weight_size + self.rnn_weight_size + self.control_weight_size + self.classifier_weight_size

    def init_networks(self):
        self.rnn_model = Sequential()
        self.rnn_model.add(SimpleRNN(self.rnn_layers[0], activation=self.rnn_activation,
                                     input_shape=(self.rnn_timesteps, self.k_size ** 2), return_sequences=True))
        self.rnn_model.add(SimpleRNN(self.rnn_layers[1], activation=self.rnn_activation))
        # self.rnn_model.summary()

        self.control_model = Sequential([Dense(units=self.control_output, input_dim=self.rnn_layers[-1])])
        # self.control_model.summary()
        self.classifier_model = Sequential([Dense(units=self.nr_of_classes, input_dim=self.rnn_layers[-1])])
        # self.classifier_model.summary()
        self.sess.run(K.tf.global_variables_initializer())

    def init_image_loader(self):
        self.train_x, self.train_y, self.test_x, self.test_y = mnist_loader.load(self.path_to_images)
        middle = math.sqrt(len(self.train_x[0])) / 2
        self.lattice = [middle, middle]

    # If init_kernel, we ignore the input and set the kernel positions
    def set_weights(self, weights):
        with self.sess.as_default():
            k, r, co, cl = self.kernel_weight_size, self.rnn_weight_size, self.control_weight_size, self.classifier_weight_size
            # 3, because x, y, std
            self.kernel_weights = np.reshape(weights[:k], (3, -1))
            w1 = k

            self.rnn_weights = []
            for w in self.rnn_model.get_weights():
                self.rnn_weights.append(np.reshape(weights[w1:w1+w.size], w.shape))
                w1 += w.size

            self.control_weights = []
            for w in self.control_model.get_weights():
                self.control_weights.append(np.reshape(weights[w1:w1+w.size], w.shape))
                w1 += w.size
            self.classifier_weights = []
            for w in self.classifier_model.get_weights():
                self.classifier_weights.append(np.reshape(weights[w1:w1+w.size], w.shape))
                w1 += w.size
            self.rnn_model.set_weights(self.rnn_weights)
            self.control_model.set_weights(self.control_weights)
            self.classifier_model.set_weights(self.classifier_weights)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_weights_size(self):
        return self.weights_size

    def set_logging(self, logging):
        self.logging = logging

    def classify(self, features):
        pass

    def get_score(self):
        return self.accuracy

    def train(self, epoch=None):
        true_positives = 0
        with self.sess.as_default():
            indices = range((epoch * self.batch_size) % len(self.train_x), ((epoch + 1) * self.batch_size ) % len(self.train_x))
            for i in indices:
                img = self.train_x[i]
                for n in range(self.rnn_timesteps):
                    k = self.kernel_weights

                    glimpse_ = GlimpseGenerator().get_glimpse(img, self.lattice[0], self.lattice[1], k[0], k[1], k[2])
                    # print("Glimpse:")
                    # print(glimpse_)
                    K.set_value(self.glimpse, glimpse_.reshape((1, 1, self.k_size ** 2)))
                    # Get the RNN params to feed to control or classifier network
                    rnn_out = self.rnn_model.call(self.glimpse)
                    # print("RNN weights:")
                    # print(rnn_out.eval())
                    control_out = self.control_model.call(rnn_out)
                    # print(type(control_out))
                    control_out = control_out.eval()
                    class_out = self.classifier_model.call(rnn_out).eval()
                    self.lattice[0] = control_out[0][0]
                    self.lattice[1] = control_out[0][1]
                    # print(class_out)
                    # print(control_out)
                    true_positives += np.argmax(class_out) == self.train_y[i]
        # K.clear_session()
        # TODO - simplest scoring right now - we probably want to change this to reward guessing quicker
        self.accuracy = true_positives / (self.batch_size * self.rnn_timesteps)
        # print("acc: {}".format(self.accuracy))

    def test(self):
        pass

    def visualize(self, epoch, res_directory=None, filename=None):
        scale = 20
        img = np.zeros((scale *self.image_size, scale*self.image_size, 3) ,np.uint8)
        for i in self.kernel_weights.T:
            img = cv2.circle(img, (int((self.image_size / 2 - int(i[0])) * scale), int((self.image_size / 2 - int(i[1])) * scale)), abs(int(i[2] * scale)), (0, 0, 255), -1)
        if filename is None:
            filename = res_directory + "lattice-epoch_{}-{}.png".format(epoch, str(time.time())[-5:])
        cv2.imwrite(filename, img)
        # cv2.waitKey(0)


if __name__ == '__main__':

    # r = 1
    # lattice = create_lattice(14, 14, 2 * r, 12)
    # img = np.zeros((28, 28, 1), dtype=np.uint8)
    # for i, j in lattice:
    #     img = draw_circle(img, i, j, r)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)

    Model().init_image_loader()