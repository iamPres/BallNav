import math
import uuid
import numpy as np

class NeuralNetwork:

    def __init__(self):
        self.nodesByLayer = [4, 5, 2]
        self.layerNumber = len(self.nodesByLayer)-2
        self.fitness = 0
        self.tag = uuid.uuid1()
        self.network = []

        for i in range(self.layerNumber):
            #layer = np.subtract(np.dot(np.random.random_sample((self.nodesByLayer[i], self.nodesByLayer[i+1])), 2), 1)
            layer = np.zeros((self.nodesByLayer[i], self.nodesByLayer[i+1]))
            self.network.append(layer)

    def linear(self, x, layerIndex):
        input_matrix = np.transpose([x] * self.network[layerIndex].shape[1])
        weighted_inputs = input_matrix * self.network[layerIndex]
        output = np.tanh(np.sum(weighted_inputs, axis=0))
        return output

    def forward(self, x):
        x = np.array(x).flatten()
        for layer in range(len(self.network)):
            x = self.linear(x, layer)
        return x
