import numpy as np

class NeuralNetwork():
    def __init__(self, num_inputs, num_outputs=1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []

    class Layer():
        def __init__(self, num_neurons, num_inputs, activation_function='sigmoid'):
            self.num_neurons = num_neurons
            self.num_inputs = num_inputs
            self.weights = 2 * np.random.rand(num_neurons, num_inputs) -1
            self.activation_function = self.activation_function(activation_function)
            self.output = np.zeros(self.num_neurons)

        def activation_function(self, activation_function):
            if activation_function == 'sigmoid':
                return self._sigmoid

        def forward(self, x):
            self.output = self.activation_function(np.dot(x, self.weights.T))
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

    def backpropagation(self):


if __name__ == "__main__":
    x = (0, 0, 1):
    nn = NeuralNetwork(3)
    nn.add_layer(4)
    nn.add_layer(2)
    nn.add_layer(1)
    print(nn.forward(x))

