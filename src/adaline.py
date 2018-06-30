class Adaline():
    def __init__(self, input_shape, learning_rate=0.1, epochs=50):
        self.weights = np.zeros(input_shape + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, X):
        return sigmoid(self.net_input(X))

    def predict(self, features):
        x = np.insert(features, 0, 1)
        x = self.weights.T.dot(x)
        x = self.activation_function(x)

    def fit(self, features, labels):
        self.cost_ = []
        for i in range(self.epochs):
            output = self.activation_function(features)
            errors = (y - output)
            errors = (erros**2).sum() / 2.0
            self.weights[1:] += self.learning_rate * features.T.dot(errors)
            self.weights[0] += self.learning_rate * errors
            cost = errors
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
