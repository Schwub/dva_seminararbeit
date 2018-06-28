from numpy import random

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.weights = 2 * random.random([number_of_inputs_per_neuron, number_of_neurons]) - 1
