import numpy as np
import scipy as sp
import gzip
import pickle
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from sklearn.datasets import make_moons
# from PIL import Image
# from tensorflow.keras import datasets

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

class Conv2d:
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
        self.input_data = []
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
        self.input_data = X
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

        # Recalculate the gradient which will be propagated back to prev. layer
        delta = self.W_col.T.dot(delta)
        # Reshape from column shape to image shape
        da_dz = self.grad_activation(self, self.output)
        delta = column_to_image(delta,
                                self.input_data.shape,
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding) * da_dz
        return delta, d_w, d_b

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class Flatten:
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

    def __init__(self, input_shape: list, n_nodes: int, activation_function: str):
        # Create Attributes
        self.input_data = []
        self.n_nodes = n_nodes
        self.n_dim, self.n_samples = input_shape  # columns indicate the number of training samples
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
        self.input_data = input_data
        self.z = np.dot(self.W, input_data) + self.b
        self.a = self.forward_activation(self.z)
        return self.a

    def backward_pass(self, delta):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights (includes activation layer hence da/dz)
            da_dz = self.grad_activation(self.z)
            delta = delta * da_dz
            d_w = np.dot(delta, self.input_data.T)
            d_b = np.sum(delta, axis=1, keepdims=True)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        delta = np.dot(W.T,delta)
        # np.multiply(dJ_da, da_dz)
        return delta, d_w, d_b

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


class Network:
    """ Class contains layers"""

    def __init__(self, layers: list, raw_data: np.ndarray, output_targets: np.array, lambd: float, learning_rate: float, n_epoch: int, n_batch_size: int, loss_function: str):
        self.layers = layers  # This is a list of layer classes
        self.raw_data = raw_data
        self.n_dim, self.n_samples = raw_data.shape  # columns indicate the number of training samples
        self.lambd = lambd # Regularization
        self.loss_function = loss_function
        self.output_values = []
        self.matrix_cost = []
        self.output_targets = output_targets
        self.learning_rate = learning_rate
        self.dJ_da = []
        self.n_epoch = n_epoch
        self.n_batch_size = n_batch_size
        self.batched_input = []
        self.batched_output = []
        self.n_batches = []

    def forward_pass(self, input_data):
        output_values = []
        for ilayer, layer in enumerate(self.layers):
            if ilayer == 0:
                output_values = self.layers[ilayer].forward_propagation(input_data)
            else:
                output_values = self.layers[ilayer].forward_propagation(output_values)
        self.output_values = output_values
        return self.output_values

    def dJ_dw_diff(self, ilayer_w, j, k, offset):
        temp_outputs = self.forward_pass_offset(ilayer_w, j, k, offset)
        delta_J = self.calculate_cost_pred(temp_outputs) - self.calculate_cost_pred(self.output_values)
        delta_w = offset
        return delta_J/delta_w

    def dJ_db_diff(self, ilayer_b, j, b_offset):
        temp_outputs = self.forward_pass_boffset(ilayer_b, j, b_offset)
        delta_J = self.calculate_cost_pred(temp_outputs) - self.calculate_cost_pred(self.output_values)
        delta_b = b_offset
        return delta_J/delta_b

    def calculate_matrix_cost(self):
        # self.output_values - self.output_targets
        # (((1-self.output_targets)*np.log(1-self.output_values)+self.output_targets*np.log(self.output_values)))
        self.matrix_cost = - np.multiply(self.output_targets, np.log(self.output_values))

    def calculate_total_cost(self):
        # - np.sum(((1-self.output_targets)*np.log(1-self.output_values)+self.output_targets*np.log(self.output_values)))/self.n_samples
        return - (1/self.n_samples)*np.sum(np.multiply(self.output_targets, np.log(self.output_values)))

    def calculate_cost_pred(self, input_predictions, input_targets):
        self.lambd_cost = 0
        if self.lambd > 0.0:
            for ilayer, layer in enumerate(self.layers):
                self.lambd_cost = np.sum(self.layers[ilayer].W**2)
        self.lambd_cost *= self.lambd / (2 * self.n_samples)
        # - np.sum(((1-self.output_targets)*np.log(1-self.output_values)+self.output_targets*np.log(self.output_values)))/self.n_samples
        if self.loss_function == "cross_entropy":  # Binary classification
            return - (1/self.n_samples) * np.sum(np.nan_to_num(-input_targets*np.log(input_predictions)-(1-input_targets)*np.log(1-input_predictions))) + self.lambd_cost
        elif self.loss_function == "mse":
            return np.sqrt( (1.0 / self.n_samples) * np.sum((input_predictions - input_targets)**2)) + self.lambd_cost
        elif self.loss_function == "log_likelihood":  # Multi-class
            return - (1/self.n_samples) * np.sum(np.log(input_predictions)) + self.lambd_cost

    def calculate_loss_gradient(self, predictions, output_targets):
        delta = output_targets - predictions
        return delta

    def derivative_cost(self, predictions, targets):
        #  (self.matrix_cost - self.output_targets)/(self.matrix_cost - np.multiply(self.matrix_cost, self.matrix_cost))
        """Derivative of cost relative to a or last layer's activation function, softmax usually dJ_da"""
        #https://condor.depaul.edu/~ntomuro/courses/578/notes/3-Backprop-More.pdf
        #return np.divide(-self.output_targets[:, isample][np.newaxis].transpose(), predictions)
        if self.loss_function == "cross_entropy":  # Binary classification
            return -np.divide(predictions - targets, predictions * (1-predictions))
        elif self.loss_function == "mse":
            return -(targets - predictions)
        elif self.loss_function == "log_likelihood":  # Multi-class
            return -np.divide(predictions - targets, predictions * (1-predictions))


    # an auxiliary function that converts probability into class
    def convert_prob_into_binary(self, predictions, threshold):
        predictions_binary = np.copy(predictions)
        predictions_binary[predictions_binary > threshold] = 1
        predictions_binary[predictions_binary <= threshold] = 0
        return predictions_binary

    def get_accuracy(self, predictions):
        prediction_binary = self.convert_prob_into_binary(predictions, 0.5)
        return (prediction_binary == self.output_targets).all(axis=0).mean()

    def backpropagation(self, input_data, output_targets):
        n_samples_per_batch = input_data.shape[1]
        predictions = self.forward_pass(input_data)
        # First calculate error (self.delta) of output layer
        delta = self.calculate_loss_gradient( predictions, output_targets)
        L = len(self.layers)
        for iLayer in range(L-1, -1, -1):
            delta, d_w, d_b = self.layers[iLayer].backward_pass(delta)
            # Update weights
            self.layers[iLayer].W += (self.learning_rate) * d_w / n_samples_per_batch
            self.layers[iLayer].b += (self.learning_rate) * d_b / n_samples_per_batch

    def create_minibatches(self):
        if self.n_samples % self.n_batch_size != 0:
            print("Warning: create_minibatches(): Batch size {0} does not evenly divide the number of examples {1}.".format(self.n_batch_size, self.n_samples))
        batched_input = []
        batched_output = []
        idx = 0
        while idx + self.n_batch_size <= self.n_samples:
            batched_input.append(self.raw_data[:, idx:idx+self.n_batch_size])
            batched_output.append(self.output_targets[:, idx:idx+self.n_batch_size])
            idx += self.n_batch_size

        return batched_input, batched_output

    def plot_dec_bound(self, X, iepoch):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = self.forward_pass(np.c_[xx.ravel(), yy.ravel()].T)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[0, :], X[1, :], c=colour, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.title('Decision Boundary - L2 Regularization 0.05 Epoch {0}'.format(iepoch))
        plt.savefig('testplot'+str(iepoch)+'.png')
        #plt.show()
        return

    def training(self):
        #if self.n_batch_size > 1:
        self.batched_input, self.batched_output = self.create_minibatches()
        self.n_batches = len(self.batched_input)
        # images = []
        # icount=0
        for iepoch in range(self.n_epoch):
            for ibatch in range(self.n_batches):
                self.backpropagation(self.batched_input[ibatch], self.batched_output[ibatch])
            predictions = self.forward_pass(self.raw_data)
            accuracy = self.get_accuracy(predictions)
            losses = self.calculate_cost_pred(predictions, self.output_targets)
            print("epoch", iepoch,"loss", losses, "accuracy", accuracy)
        #     icount += 1
        #     if icount > 5:
        #         icount = 0
        #         self.plot_dec_bound(self.raw_data, iepoch)
        #         ims = Image.open('testplot' + str(iepoch) + '.png')
        #         images.append(ims)
        # images[0].save('regu_0_05.gif',
        #                save_all=True, append_images=images[1:], optimize=False, duration=400, loop=0)




#Load the MNIST data
def load_data_shared(filename="C:/Workspaces/NeuralNetworkFramework/data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


training_data, validation_data, test_data = load_data_shared()
# Only 5000 samples otherwise running out of memory
training_data_input = training_data[0][0:50000].transpose()
# One-hot encoding the outputs
training_data_outputs = np.zeros((10, training_data_input.shape[1]))
for iPos in range(training_data_input.shape[1]):
    training_data_outputs[training_data[1][iPos]][iPos] = 1


layer1 = Dense(training_data_input.shape, 200, "sigmoid")
layer2 = Dense(layer1.get_output_size(), 80, "sigmoid")
layer3 = Dense(layer2.get_output_size(), 10, "sigmoid")

layer_list = [layer1, layer2, layer3]

NN = Network(layer_list, training_data_input, training_data_outputs, lambd=0.02, learning_rate=0.5, n_epoch=10, n_batch_size=10, loss_function="cross_entropy")

NN.training()




# # https://www.tensorflow.org/tutorials/images/cnn
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#
# train_images = np.asarray(train_images)
# train_labels = np.asarray(train_labels)
# test_images = np.asarray(test_images)
# test_labels = np.asarray(test_labels)
#
# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0


# Tensorflow layer
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))