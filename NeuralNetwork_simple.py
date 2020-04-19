import numpy as np
import scipy as sp
import gzip
import pickle



class DenseLayer:
    """Basic definition of layer for neural networks"""
    """ Only for one sample to simplify at first"""

    def __init__(self, input_data: np.ndarray, n_nodes: int, activation_function: str):
        # Create Attributes
        self.input_data = []
        self.n_nodes = n_nodes
        self.n_dim, self.n_samples = input_data.shape  # columns indicate the number of training samples
        self.activation_function = activation_function

        self.z = np.zeros((self.n_nodes, 1))
        self.W = np.random.uniform(-1, 1, size=(self.n_nodes, self.n_dim))
        self.b = np.repeat(np.random.uniform(0.0001, 0.1, size=(self.n_nodes, 1)), 1, axis=1) # Columns need to be similar
        self.a = np.zeros((self.n_nodes, 1))
        self.da_dz = []
        self.dJ_dw = np.zeros((self.n_samples, self.n_nodes, self.n_dim))
        self.dJ_db = np.zeros((self.n_nodes, self.n_samples ))
        self.dJ_dw_total = np.zeros((self.n_nodes, self.n_dim))
        self.dJ_db_total = np.zeros((self.n_nodes, 1))
        self.delta = []

    def forward_propagation(self):
        self.z = np.matmul(self.W, self.input_data) + self.b
        self.a = self.forward_activation(self.z)
        return self.a

    def forward_activation(self, X):
        if self.activation_function == "sigmoid":
            # Prevent overflow.
            X = np.clip(X, -500, 500)
            return 1.0 / (1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0, X)
        elif self.activation_function == "softmax":
            # stable version is np.exp(X - np.max(X))/ np.sum(np.exp(X), axis=0)
            return np.exp(X) / np.sum(np.exp(X), axis=0)

    def grad_activation(self, X):
        """Note that inputs are the results of the activation function here
        for example: X = sigmoid(z)"""
        if self.activation_function == "sigmoid":
            self.da_dz = X * (1 - X)
        elif self.activation_function == "tanh":
            self.da_dz =  (1 - np.square(X))
        elif self.activation_function == "relu":
            self.da_dz =  1.0 * (X > 0)
        elif self.activation_function == "softmax":
            A = self.forward_activation(X)
            # Usually softmax is in the output layer hence n_nodes = n_outputs
            # Each sample has its own Jacobian size(n_outputs x n_outputs)
            jacobian_a = np.zeros(( self.n_nodes, self.n_nodes))
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i == j:
                        jacobian_a[i,j] = A[i] * (1 - A[i])
                    else:
                        jacobian_a[i,j] = -A[i] * A[j]
            self.da_dz = jacobian_a
        return self.da_dz

    def get_output_size(self):
        return np.array([self.n_nodes, self.n_samples])



class Network:
    """ Class contains layers"""

    def __init__(self, layers: list, input_data: np.ndarray, output_targets: np.array, learning_rate: float, n_epoch: int):
        self.layers = layers  # This is a list of layer classes
        self.input_data = input_data
        self.n_dim, self.n_samples = input_data.shape  # columns indicate the number of training samples
        self.output_values = []
        self.matrix_cost = []
        self.output_targets = output_targets
        self.n_classes = output_targets.shape[1]
        self.learning_rate = learning_rate
        self.dJ_da = []
        self.n_epoch = n_epoch

    def forward_pass(self, isample):
        output_values = []
        for ilayer, layer in enumerate(self.layers):
            if ilayer == 0:
                self.layers[ilayer].input_data = self.input_data[:, isample][np.newaxis].transpose()
            else:
                self.layers[ilayer].input_data = output_values
            output_values = self.layers[ilayer].forward_propagation()
        return output_values

    def calculate_total_cost(self, predictions, isample):
        return -np.sum(np.multiply(self.output_targets[:, isample], np.log(predictions)))

    def derivative_cost(self, predictions, isample):
        #  (self.matrix_cost - self.output_targets)/(self.matrix_cost - np.multiply(self.matrix_cost, self.matrix_cost))
        """Derivative of cost relative to a or last layer's activation function, softmax usually dJ_da"""
        return np.divide(-self.output_targets[:, isample][np.newaxis].transpose(), predictions)

    def backpropagation(self):
        for isample in range(self.n_samples):
            predictions = self.forward_pass(isample)
            # First calculate error (self.delta) of output layer
            L = len(self.layers)-1
            # Number of outputs equal to number of nodes of the output layer
            self.layers[L].delta = np.zeros((self.layers[L].n_nodes, 1))
            dCost_da = self.derivative_cost(predictions, isample) # Calculates dJ_da of output layer
            da_dz = self.layers[L].grad_activation(self.layers[L].z) # Calculates da_dz
            if self.layers[L].activation_function == "softmax":
                # Softmax case is a bit special because Jacobian
                for sample in range(self.n_samples):
                    self.layers[L].delta = np.matmul(da_dz, dCost_da)
            else:
                self.layers[L].delta = np.multiply(da_dz, dCost_da)

            delta_test = predictions - self.output_targets[:, isample][np.newaxis].transpose()

            ## For each sample we calculate the dJ_dw and dJ_db
            ## size(dJ_dw) = (samples, size of W matrix)
            #self.layers[L].dJ_dw = np.zeros((self.layers[L].n_nodes, self.layers[L].n_dim))
            ## Preferred matrix arranged in columns for each sample so no need to loop through it
            ## self.layers[L].dJ_db = np.zeros((self.layers[L].n_nodes,self.n_samples ))
            #self.layers[L].dJ_db = self.layers[L].delta
            #self.layers[L].dJ_dw[:, :] = np.matmul(np.transpose(self.layers[L].delta[:][np.newaxis]), self.layers[L-1].a[:][np.newaxis])

            # Loop through the rest of layers from L-1 to 0
            for iLayer in range(L-1,-1,-1): # range(start,stop,step)
                da_dz = self.layers[iLayer].grad_activation(self.layers[iLayer].z)  # Calculates da_dz
                self.layers[iLayer].delta = np.multiply(np.matmul(self.layers[iLayer+1].W.transpose(), self.layers[iLayer+1].delta), da_dz)

            # Loop through all layers from L to 0
            for iLayer in range(L, -1, -1):
                # For each sample we calculate the dJ_dw and dJ_db
                # size(dJ_dw) = (samples, size of W matrix)
                self.layers[iLayer].dJ_dw = np.zeros((self.layers[iLayer].n_nodes, self.layers[iLayer].n_dim))
                # Preferred matrix arranged in columns for each sample so no need to loop through it
                self.layers[iLayer].dJ_db = np.zeros((self.layers[iLayer].n_nodes, 1))
                self.layers[iLayer].dJ_db = self.layers[iLayer].delta
                if iLayer - 1 < 0:  # In this case the iLayer-1 is equal to the input layer
                    self.layers[iLayer].dJ_dw[:, :] = np.matmul(self.layers[iLayer].delta[:][np.newaxis], self.input_data[:,isample][np.newaxis])
                else:
                    self.layers[iLayer].dJ_dw[:, :] = np.matmul(self.layers[iLayer].delta[:][np.newaxis], self.layers[iLayer-1].a[:].transpose())

                self.layers[iLayer].dJ_dw_total = self.layers[iLayer].dJ_dw_total + self.layers[iLayer].dJ_dw
                self.layers[iLayer].dJ_db_total = self.layers[iLayer].dJ_db_total + self.layers[iLayer].dJ_db


        # For each layer update weights. The correction of the wights is the average of the dJ/dw and dJ/db for all
        # the training samples
        for ilayer, layer in enumerate(self.layers):
            self.layers[ilayer].W = self.layers[ilayer].W - (self.learning_rate/self.n_samples)*self.layers[ilayer].dJ_dw_total
            self.layers[ilayer].b = self.layers[ilayer].b - (self.learning_rate/self.n_samples)*self.layers[ilayer].dJ_db_total
        sample_cost = self.calculate_total_cost(predictions, 0)
        print(sample_cost)
    def training(self):
        for iepoch in range(self.n_epoch):
            self.backpropagation()



# Load the MNIST data
def load_data_shared(filename="C:/Workspaces/NeuralNetworkFramework/data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


training_data, validation_data, test_data = load_data_shared()


# Only 5000 samples otherwise running out of memory
training_data_input = training_data[0][0:5000].transpose()
# One-hot encoding the outputs
training_data_outputs = np.zeros((10, training_data_input.shape[1]))
for iPos in range(training_data_input.shape[1]):
    training_data_outputs[training_data[1][iPos]][iPos] = 1



layer1 = DenseLayer(training_data_input, 32, "sigmoid")
layer2 = DenseLayer(np.zeros( layer1.get_output_size()), 16, "sigmoid")
layer3 = DenseLayer(np.zeros( layer2.get_output_size()), 10, "softmax")


layer_list = [layer1, layer2, layer3]

NN = Network(layer_list, training_data_input, training_data_outputs, learning_rate=0.01, n_epoch=100)

NN.training()

print('Hello')