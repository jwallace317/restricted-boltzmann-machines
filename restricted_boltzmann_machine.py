"""
Restricted Boltzmann Machine Module
"""

# import necessary modules
import numpy as np


class RBM():
    """
    Restricted Boltzmann Machine Class
    """

    def __init__(self, v_dim, h_dim):

        # set visible and hidden dimensions
        self.v_dim = v_dim
        self.h_dim = h_dim

        # initialize weights matrix
        self.w = np.zeros((v_dim, h_dim))

        # initialize hidden units bias vector
        self.h_bias = np.zeros((1, h_dim))

        # initialize visible units bias vector
        self.v_bias = np.zeros((v_dim, 1))

    def sample_hidden(self, visible):

        # reshape visible input
        visible = visible.reshape((1, self.v_dim))

        # calculate linear combination
        linear = np.dot(visible, self.w) + self.h_bias

        # calculate sigmoid activation
        sigmoid = 1 / (1 + np.exp(-linear))

        # sample hidden units
        random = np.random.uniform(size=(1, self.h_dim))
        hidden = np.where(random < sigmoid, 1, 0)

        return hidden

    def sample_visible(self, hidden):

        # reshape hidden input
        hidden = hidden.reshape((self.h_dim, 1))

        # compute linear combination
        linear = np.dot(self.w, hidden) + self.v_bias

        # compute sigmoid activation
        sigmoid = 1 / (1 + np.exp(-linear))

        # sample visible units
        random = np.random.uniform(size=(self.v_dim, 1))
        visible = np.where(random < sigmoid, 1, 0)

        return visible

    def train(self, visible, lr=0.01):
        """
        Train Method
        """

        # reshape visible input
        visible = visible.reshape((1, self.v_dim))

        # sample hidden units
        hidden = self.sample_hidden(visible)

        # compute positive gradient
        positive_gradient = np.outer(visible, hidden)

        # sample visible units
        _visible = self.sample_visible(hidden)

        # sample hidden units
        _hidden = self.sample_hidden(_visible)

        # compute negative gradient
        negative_gradient = np.outer(_visible, _hidden)

        # update weights
        self.weights += lr * (positive_gradient - negative_gradient)

        # update hidden bias
        self.h_bias += lr * (hidden - _hidden)

        # update visible bias
        self.v_bias += lr * (visible - _visible)

    def test(self, features, targets):
        """
        Test Method
        """

        return 0
