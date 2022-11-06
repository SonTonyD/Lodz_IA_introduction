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
        self.weights = np.array([])

        for i in range(nbNeuron):
            self.neurons.append( Neuron(self.lr, self.actFunc, 0) )

    def initWeight(self, input):
        for neuron in self.neurons:
            neuron.initNeuronWeight(input)

    def setInput(self, input):
        self.input = input
    
    def feedNeurons(self):
        for neuron in self.neurons:
            neuron.setInput(self.input)
    
    def computeOutputs(self):
        for neuron in self.neurons:
            self.output = np.append(self.output, neuron.prediction())




'''
input = np.array([5,4])
actFunc = 'heaviside'
nbNeuron = 2
lr = 0.1
'''

'''
l1 = Layer(actFunc, nbNeuron, lr)
l1.initWeight(input)


l1.setInput(input)
l1.feedNeurons()
l1.computeOutputs()
print(l1.output)


l2 = Layer(actFunc, 3, lr)
l2.initWeight(l1.output)

l2.setInput(l1.output)
l2.feedNeurons()
l2.computeOutputs()
print(l2.output)


l3 = Layer(actFunc, 2, lr)
l3.initWeight(l2.output)

l3.setInput(l2.output)
l3.feedNeurons()
l3.computeOutputs()
print(l3.output)
'''

'''
input = np.array([-5,-4])
l1 = Layer(actFunc, 2, lr)
l2 = Layer(actFunc, 3, lr)
l3 = Layer(actFunc, 2, lr)

layers = [l1, l2, l3]
currentInput = input

#init weight of all layers
for layer in layers:
    layer.initWeight(currentInput)

#procede to a feedforward 
for layer in layers:
    layer.setInput(currentInput)
    layer.feedNeurons()
    layer.computeOutputs()
    currentInput = layer.output

print("neuron output: ", currentInput)

'''
