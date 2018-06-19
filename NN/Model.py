import math
import numpy as np

import keras.backend as K
import keras
from keras import Sequential, Input
from keras.layers import RNN, Dense, SimpleRNN, Masking, LSTM, TimeDistributed
from keras.optimizers import sgd
from tensorflow.python.ops.rnn_cell_impl import BasicRNNCell
from image_processing import mnist_loader
from image_processing.GlimpseGenerator import GlimpseGenerator

class Model():

    def __init__(self, logging = False):
        self.logging = logging
        self.rnn_timesteps = 5
        self.k_size = 12
        self.rnn_layers = [500, 500]
        self.rnn_activation = 'relu'
        self.path_to_images = "data/mnist/mnist.pkl"

        self.batch_size = 32
        self.init_weight_sizes()
        self.init_networks()
        self.init_image_loader()

    def init_weight_sizes(self):
        # 4 because we need x, y coordinates and std in both directions
        self.kernel_weight_size = 4 * self.k_size ** 2
        self.rnn_weight_size = self.k_size ** 2 * self.rnn_layers[0] + self.rnn_layers[0] * self.rnn_layers[1]
        # x, y, (zoom) of full lattice
        self.control_weight_size = self.rnn_layers[1] * 3
        self.classifier_weight_size = self.rnn_layers[1] * 10
        self.weights_size = self.kernel_weight_size + self.rnn_weight_size + self.control_weight_size + self.classifier_weight_size

    def init_networks(self):
        self.rnn_model = Sequential()
        self.rnn_model.add(SimpleRNN(self.rnn_layers[0], activation=self.rnn_activation, input_shape=(self.rnn_timesteps, self.k_size ** 2), return_sequences=True))
        self.rnn_model.add(SimpleRNN(self.rnn_layers[1], activation=self.rnn_activation))
        self.rnn_model.summary()
        # TODO -  ADD BIAS!
        self.control = Sequential([Dense(units=3, input_dim=self.rnn_layers[-1], use_bias=False)])
        self.control.summary()
        # TODO -  ADD BIAS!
        self.classifier = Sequential([Dense(units=10, input_dim=self.rnn_layers[-1], use_bias=False)])
        self.classifier.summary()

    def init_image_loader(self):
        self.train_x, self.train_y, self.test_x, self.test_y = mnist_loader.load(self.path_to_images)
        middle = math.sqrt(len(self.train_x[0])) / 2
        self.lattice = [middle, middle]

    def set_weights(self, weights):
        k, r, co, cl = self.kernel_weight_size, self.rnn_weight_size, self.control_weight_size, self.classifier_weight_size
        # 4, because x, y, std x, std y
        self.kernel_weights = np.reshape(weights[:k], (4, -1))
        rnn_weights = weights[k: k + r]
        self.rnn_weights = []
        w = self.rnn_model.get_weights()
        self.rnn_weights.append(np.reshape(rnn_weights[:w[0].size], w[0].shape))
        self.rnn_weights.append(np.reshape(rnn_weights[w[0].size:], w[1].shape))
        self.control_weights = weights[k + r : k + r + co].reshape(500, 3)
        self.classifier_weights = weights[k + r + co : k + r + co + cl].reshape(500, -1)

        self.rnn_model.set_weights(self.rnn_weights)
        w = self.control.get_weights()
        self.control.set_weights([self.control_weights])
        self.classifier.set_weights([self.classifier_weights])

    def get_weights_size(self):
       return self.weights_size

    def set_logging(self, logging):
        self.logging = logging

    def classify(self, features):
        pass

    def get_score(self):
        return self.accuracy

    def train(self):
        glimpse = K.variable(np.zeros((1, 1, 144)))
        true_positives = 0
        with K.tf.Session() as sess:
            sess.run(K.tf.global_variables_initializer())

            for i in range(self.batch_size):
                # print("img nr {}".format(i))
                img = self.train_x[i]
                for n in range(self.rnn_timesteps):
                    k = self.kernel_weights

                    glimpse_ = GlimpseGenerator().get_glimpse(img, self.lattice[0], self.lattice[1], k[0], k[1], k[2], k[3])
                    K.set_value(glimpse, glimpse_.reshape((1,1,144)))
                    # Get the RNN params to feed to control or classifier network
                    rnn_out = self.rnn_model.call(glimpse)
                    # print(rnn_out.eval())
                    control_out = self.control.call(rnn_out)
                    # print(type(control_out))
                    control_out = control_out.eval()
                    class_out = self.classifier.call(rnn_out).eval()
                    self.lattice[0] = control_out[0][0]
                    self.lattice[1] = control_out[0][1]
                    # print(class_out)
                    # print(np.argmax(class_out))
                    # print(control_out)
                    true_positives += np.argmax(class_out) == self.train_y[i]
        # TODO - simplest scoring right now - we probably want to change this to reward guessing quicker
        self.accuracy = true_positives / (self.batch_size * self.rnn_timesteps)
        # print("acc: {}".format(self.accuracy))

    def test(self):
        pass


if __name__ == '__main__':
    feats_dim = 144
    hidden_dims = [500, 500]
    model = Sequential()
    model.add(SimpleRNN(500, return_sequences=True, consume_less='gpu', input_shape=(5, 144)))
    model.add(SimpleRNN(500, consume_less='gpu'))

    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.call(np.zeros((1, 1, 144)))