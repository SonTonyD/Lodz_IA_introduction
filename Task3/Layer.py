import math
import numpy as np
from Neuron_3 import Neuron
class Layer:
    def __init__(self, input, actFunc, nbNeuron, lr):
        self.input = input
        self.actFunc = actFunc
        self.nbNeuron = nbNeuron
        self.lr = lr
        self.neurons = []

        for i in range(nbNeuron):
            self.neurons.append( Neuron(self.lr, self.actFunc, 0) )
    
    def feedNeurons(self):
        pass


input = np.array([5,4])
actFunc = 'heaviside'
nbNeuron = 2
lr = 0.1

l1 = Layer(input, actFunc, nbNeuron, lr)
print(l1.neurons)