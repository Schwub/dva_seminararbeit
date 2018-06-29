import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork():
    def __init__(self, num_inputs, num_outputs=1, lr=0.1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
        self.layers = []

    class Layer():
        def __init__(self, num_neurons, num_inputs):
            self.num_neurons = num_neurons
            self.num_inputs = num_inputs
            self.weights = 2 * np.random.rand(num_inputs, num_neurons) -1
            self.bias = np.random.rand(1, num_neurons)
            self.activation_function = self._sigmoid
            self.output = np.zeros(self.num_neurons)

        def activation_function(self, activation_function):
            if activation_function == 'sigmoid':
                return self._sigmoid

        def forward(self, x):
            self.output = self.activation_function(np.dot(x, self.weights) + self.bias)
            return self.output

        def _sigmoid(self, x):
            return 1.0 / (1.0 + np.exp(-x))

    def add_layer(self, num_neurons):
        if not self.layers:
            self.layers.append(self.Layer(num_neurons, self.num_inputs))
        else:
            self.layers.append(self.Layer(num_neurons, self.layers[-1].num_neurons))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backpropagation(self, x, y):
        slope_layer3 = self._derivates_sigmoid(self.layers[2].output)
        slope_layer2 = self._derivates_sigmoid(self.layers[1].output)
        slope_layer1 = self._derivates_sigmoid(self.layers[0].output)


        error_layer3 = y - self.layers[2].output
        d_layer3 = error_layer3 * slope_layer3

        error_layer2 = np.dot(d_layer3, self.layers[2].weights.T)
        d_layer2 = error_layer2 * slope_layer2

        error_layer1 = np.dot(d_layer2, self.layers[1].weights.T)
        d_layer1 = error_layer1 * slope_layer1

        self.layers[2].weights += (self.layers[1].output.T.dot(d_layer3))
        self.layers[2].bias += np.sum(d_layer3, axis=0, keepdims=True)
        self.layers[1].weights += (self.layers[0].output.T.dot(d_layer2))
        self.layers[1].bias += np.sum(d_layer2, axis=0, keepdims=True)
        self.layers[0].weights += ((np.reshape(x, x.shape)).T.dot(d_layer1))
        self.layers[0].bias += np.sum(d_layer1, axis=0, keepdims=True)

    def _derivates_sigmoid(self, x):
        return x * (1.0 - x)

if __name__ == "__main__":
    data = load_iris()
    x = data.data[0:100]
    y = data.target[0:100]
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    print(x)
    print(y)
    y = np.reshape(y, (len(y), 1))
    nn = NeuralNetwork(4)
    nn.add_layer(5)
    nn.add_layer(3)
    nn.add_layer(1)

    for i in range(500):
        print(nn.forward(x))
        nn.backpropagation(x, y)

