import numpy as np
import scipy as sp
import gzip
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons
from PIL import Image
from tensorflow.keras import datasets

import math

# Method which calculates the padding based on the specified output shape and the
# shape of the filters
def determine_padding(filter_shape, output_shape="same"):

    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

# Reference: CS231n Stanford
def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)

# Method which turns the image shaped input to column shape.
# Used during the forward pass.
# Reference: CS231n Stanford - Implementation as Matrix Multiplication
# https://cs231n.github.io/convolutional-networks/
def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape

    pad_h, pad_w = determine_padding(filter_shape, output_shape)

    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


# Method which turns the column shaped input to image shape.
# Used during the backward pass.
# Reference: CS231n Stanford - Implementation as Matrix Multiplication
# https://cs231n.github.io/convolutional-networks/
def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.empty((batch_size, channels, height_padded, width_padded))

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    # Return image without padding
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]

class Conv2d():
    """A 2D Convolution Layer.
    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """
    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1, activation_function='ReLU'):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True
        self.activation_function = activation_function
        channels = self.input_shape[0]
        filter_height, filter_width = self.filter_shape
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.W = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.b = np.zeros((self.n_filters, 1))
        self.output = np.zeros((self.output_shape()))

    def forward_activation(self, z):
        if self.activation_function == "sigmoid":
            # Prevent overflow.
            z = np.clip(z, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))
        elif self.activation_function == "tanh":
            return np.tanh(z)
        elif self.activation_function == "ReLU":
            return np.where(z >= 0, z, 0)
        elif self.activation_function == "softmax":
            # stable version is np.exp(X - np.max(X))/ np.sum(np.exp(X), axis=0)
            return np.exp(z) / np.sum(np.exp(z), axis=0)

    def grad_activation(self, z):
        """Note that inputs are the results of the activation function here
        for example: X = sigmoid(z)"""
        if self.activation_function == "sigmoid":
            a = self.forward_activation(z)
            self.da_dz = a*(1-a)
        elif self.activation_function == "tanh":
            self.da_dz = (1 - np.square(z))
        elif self.activation_function == "ReLU":
            self.da_dz = np.where(z >= 0, 1, 0)
        elif self.activation_function == "softmax":
            a= self.forward_activation(z)
            # Usually softmax is in the output layer hence n_nodes = n_outputs
            # Each sample has its own Jacobian size(n_outputs x n_outputs)
            jacobian_a = np.zeros((z.shape[1], self.n_nodes, self.n_nodes))
            for sample in range(z.shape[1]):
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i == j:
                            jacobian_a[sample,i,j] = a[i][sample] * (1 - a[i][sample])
                        else:
                            jacobian_a[sample,i,j] = -a[i][sample] * a[j][sample]
            self.da_dz = jacobian_a
        return self.da_dz

    def forward_pass(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape((self.n_filters, -1))
        # Calculate output
        self.output = self.W_col.dot(self.X_col) + self.b
        # Reshape into (n_filters, out_height, out_width, batch_size)
        self.output = self.output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        self.output = self.output.transpose(3,0,1,2)
        return self.forward_activation(self.output)

    def backward_pass(self, delta):
        # Reshape accumulated gradient into column shape
        # delta = delta cost / delta x
        delta = delta.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        if self.trainable:
            # Take dot product between column shaped accum. gradient and column shape
            # layer input to determine the gradient at the layer with respect to layer weights
            d_w = delta.dot(self.X_col.T).reshape(self.W.shape)
            # The gradient with respect to bias terms is the sum similarly to in Dense layer
            d_b = np.sum(delta, axis=1, keepdims=True)

            # Update the layers weights
            self.W = self.W_opt.update(self.W, d_w)
            self.b = self.w0_opt.update(self.b, d_b)

        # Recalculate the gradient which will be propagated back to prev. layer
        delta = self.W_col.T.dot(delta)
        # Reshape from column shape to image shape
        da_dz = self.grad_activation(self, self.output)
        delta = column_to_image(delta,
                                self.layer_input.shape,
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding) * da_dz
        return delta

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class Flatten():
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)



class Dense:
    """Basic definition of layer for neural networks"""

    def __init__(self, raw_data: np.ndarray, n_nodes: int, activation_function: str):
        # Create Attributes
        self.raw_data = raw_data
        self.input_data = []
        self.n_nodes = n_nodes
        self.n_dim, self.n_samples = raw_data.shape  # columns indicate the number of training samples
        self.activation_function = activation_function
        self.trainable = True

        self.z = np.zeros((self.n_nodes, 1))
        self.W = np.random.randn(self.n_nodes, self.n_dim)*np.sqrt(1/((self.n_nodes + self.n_dim)))
        self.b = np.zeros((self.n_nodes, 1))  # Columns need to be similar
        self.a = np.zeros((self.n_nodes, 1))
        self.da_dz = []
        self.dJ_dw = np.zeros((self.n_nodes, self.n_dim))
        self.dJ_db = np.ones((self.n_nodes, 1))
        self.delta = []

    def forward_propagation(self, input_data):
        self.z = np.dot(self.W, input_data) + self.b
        self.a = self.forward_activation(self.z)
        return self.a

    def backward_pass(self, delta):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            d_w = self.layer_input.T.dot(delta)
            d_b = np.sum(delta, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, d_w)
            self.b = self.w0_opt.update(self.b, d_b)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        da_dz = self.grad_activation(self, self.z)
        delta = delta.dot(W.T) * da_dz
        # np.multiply(dJ_da, da_dz)
        return delta

    def forward_activation(self, z):
        if self.activation_function == "sigmoid":
            # Prevent overflow.
            z = np.clip(z, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))
        elif self.activation_function == "tanh":
            return np.tanh(z)
        elif self.activation_function == "relu":
            return np.maximum(0, z)
        elif self.activation_function == "softmax":
            # stable version is np.exp(X - np.max(X))/ np.sum(np.exp(X), axis=0)
            return np.exp(z) / np.sum(np.exp(z), axis=0)

    def grad_activation(self, z):
        """Note that inputs are the results of the activation function here
        for example: X = sigmoid(z)"""
        if self.activation_function == "sigmoid":
            a = self.forward_activation(z)
            self.da_dz = a*(1-a)
        elif self.activation_function == "tanh":
            self.da_dz = (1 - np.square(z))
        elif self.activation_function == "relu":
            self.da_dz = 1.0 * (z > 0)
        elif self.activation_function == "softmax":
            a= self.forward_activation(z)
            # Usually softmax is in the output layer hence n_nodes = n_outputs
            # Each sample has its own Jacobian size(n_outputs x n_outputs)
            jacobian_a = np.zeros((z.shape[1], self.n_nodes, self.n_nodes))
            for sample in range(z.shape[1]):
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i == j:
                            jacobian_a[sample,i,j] = a[i][sample] * (1 - a[i][sample])
                        else:
                            jacobian_a[sample,i,j] = -a[i][sample] * a[j][sample]
            self.da_dz = jacobian_a
        return self.da_dz

    def get_output_size(self):
        return np.array([self.n_nodes, self.n_samples])


class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))


    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T)
        return accum_grad





# https://www.tensorflow.org/tutorials/images/cnn
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Tensorflow layer
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))