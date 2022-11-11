import math
import numpy as np
from Neuron_3 import Neuron
class Layer:
    def __init__(self, actFunc, nbNeuron, lr, logisticParam):
        self.input = np.array([])
        self.actFunc = actFunc
        self.nbNeuron = nbNeuron
        self.lr = lr
        self.neurons = []
        self.output = np.array([])
        self.weights = np.array([])
        self.errors = np.array([])
        self.logisticParam = logisticParam

        for i in range(nbNeuron):
            self.neurons.append( Neuron(self.lr, self.actFunc, logisticParam) )

    def initWeight(self, input):
        for neuron in self.neurons:
            neuron.initNeuronWeight(input)

    def setInput(self, input):
        self.input = input
    
    def feedNeurons(self):
        for neuron in self.neurons:
            neuron.setInput(self.input)
    
    def computeOutputs(self):
        self.output = np.array([])
        for neuron in self.neurons:
            self.output = np.append(self.output, neuron.prediction())

    def computeOutputsError(self):
        self.errors = np.array([])
        for neuron in self.neurons:
            error = neuron.computeErrorOutput()
            self.errors = np.append(self.errors, error)

    def computeError(self, previousLayer):
        previousErrors = previousLayer.errors
        self.errors = np.array([])
        neuronId = 0
        for neuron in self.neurons:
            previousWeights = self.getPreviousWeight(neuronId, previousLayer)
            error = neuron.computeError(previousErrors, previousWeights)
            self.errors = np.append(self.errors, error)
            neuronId += 1

    def getPreviousWeight(self, neuronId, previousLayer):
        previousWeights = np.array([])
        for neuron in previousLayer.neurons:
            previousWeights = np.append(previousWeights, neuron.weight[neuronId+1])
        return previousWeights


    def setTarget(self, target):
        for i in range(len(self.neurons)):
            self.neurons[i].setTarget(target[i])

    def updateWeights(self):
        for neuron in self.neurons:
            neuron.updateWeights()

    def printWeights(self):
        for neuron in self.neurons:
            print(neuron.weight)




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
