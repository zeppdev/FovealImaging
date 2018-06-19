import numpy as np

import keras.backend as K
import keras
from keras import Sequential
from keras.layers import RNN, Dense, SimpleRNN


class Model():

    def __init__(self, logging = False):
        self.logging = logging

        self.k_size = 12
        self.rnn_layers = [500, 500]
        self.rnn_activation = 'relu'

        self.init_weight_sizes()
        self.init_networks()

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
        self.rnn_model.add(RNN(self.rnn_layers[0], self.rnn_activation, input_shape=(self.k_size ** 2,)))
        self.rnn_model.add(RNN(self.rnn_layers[1], self.rnn_activation))

        self.control = Sequential(Dense(output_dim=3, input_dim=self.rnn_layers[-1]))
        self.classifier = Sequential(Dense(output_dim=10, input_dim=self.rnn_layers[-1]))

    def set_weights(self, weights):
        k, r, co, cl = self.kernel_weight_size, self.rnn_weight_size, self.control_weight_size, self.classifier_weight_size
        self.kernel_weights = weights[:k]
        self.rnn_weights = np.reshape(weights[k : k + r], (len(self.rnn_layers), -1))
        self.control_weights = weights[k + r : k + r + co]
        self.classifier_weights = weights[k + r + co : k + r + co + cl]

    def get_weights_size(self):
       return self.weights_size

    def set_logging(self, logging):
        self.logging = logging

    def classify(self, features):
        pass

    def get_score(self):
        pass

    def train(self):
        pass

    def test(self):
        pass