import numpy as np

class Perceptron():
    def __init__(self, input_shape, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_shape + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return binary_step(x)

    def predict(self, features):
        #Insert Bias of 1 at Postion 0
        x = np.insert(features, 0, 1)
        x = self.weights.T.dot(x)
        x = self.activation_function(x)
        return x

    def fit(self, features, labels):
        for _ in range(self.epochs):
            for i in range(len(labels)):
                predicted_label = self.predict(features[i])
                error = labels[i] - predicted_label
                if error != 0:
                    self.update_weights(features[i], error)

    def update_weights(self, features, error):
        self.weights = self.weights + self.learning_rate * error * np.insert(features, 0, 1)

    def binary_step(x):
        return np.where(x >= 0.0, 1, 0)
