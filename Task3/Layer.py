import math
import numpy as np
from Neuron_3 import Neuron
class Layer:
    def __init__(self, actFunc, nbNeuron, lr):
        self.input = np.array([])
        self.actFunc = actFunc
        self.nbNeuron = nbNeuron
        self.lr = lr
        self.neurons = []
        self.output = np.array([])

        for i in range(nbNeuron):
            self.neurons.append( Neuron(self.lr, self.actFunc, 0) )

    def setInput(self, input):
        self.input = input
        self.input = np.insert(self.input, 0, -1)
    
    def feedNeurons(self):
        for neuron in self.neurons:
            neuron.setInput(self.input[0], self.input[1])
    
    def computeOutputs(self):
        for neuron in self.neurons:
            self.output = np.append(self.output, neuron.prediction())


input = np.array([5,4])
actFunc = 'heaviside'
nbNeuron = 2
lr = 0.1

l1 = Layer(actFunc, nbNeuron, lr)

l1.setInput(input)
l1.feedNeurons()
l1Output = l1.computeOutputs()

print(l1.output)

l2 = Layer(actFunc, 3, lr)
l2.setInput(l1Output)


