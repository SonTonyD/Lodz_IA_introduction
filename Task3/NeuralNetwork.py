import math
import random
import numpy as np
from Layer import Layer
class NeuralNetwork:
    def __init__(self, nbInputNeuron, nbHiddenNeuron, nbOutputNeuron):
        self.input = 0
        self.target = 0
        self.inputLayer = Layer('sin', nbInputNeuron, 0.1)
        self.hiddenLayer = Layer('sin', nbHiddenNeuron, 0.1)
        self.outputLayer = Layer('logistic', nbOutputNeuron, 0.1)

        self.Layers = [self.inputLayer, self.hiddenLayer, self.outputLayer]

    def setInput(self, input):
        self.input = input
    
    def initAllWeights(self):          
        currentInput = self.input
        for layer in self.Layers:
            layer.initWeight(currentInput)
            layer.setInput(currentInput)
            layer.feedNeurons()
            layer.computeOutputs()
            currentInput = layer.output

    def feedforward(self):
        currentInput = self.input
        for layer in self.Layers:
            layer.setInput(currentInput)
            layer.feedNeurons()
            layer.computeOutputs()
            currentInput = layer.output
    
    def setTarget(self, target):
        self.target = target
        self.outputLayer.setTarget(target)

    def backpropagation(self):
        self.Layers[len(self.Layers)-1].computeOutputsError()
        previousLayer = self.Layers[len(self.Layers)-1]
        for i in range(len(self.Layers)-1):
            self.Layers[len(self.Layers)-2-i].computeError(previousLayer)
            previousLayer = self.Layers[len(self.Layers)-2-i]

    def updateWeight(self):
        for layer in self.Layers:
            layer.updateWeights()

    def printWeights(self):
        for layer in self.Layers:
            layer.printWeights()





input = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[1,0],[0,1],[0,1],[1,0]])



nn = NeuralNetwork(2, 3, 2)

nn.setInput(input[0])
nn.setTarget(target[0])
nn.initAllWeights()
nn.printWeights()
for i in range(4000):
    index = random.randint(0,3)
    nn.setInput(input[index])
    nn.setTarget(target[index])
    nn.feedforward()
    nn.backpropagation()
    nn.updateWeight()

nn.printWeights()

print("result")
for i in range(4):
    nn.setInput(input[i])
    nn.feedforward()
    print(nn.outputLayer.output)


'''
nn.outputLayer.computeOutputsError()
nn.hiddenLayer.computeError(nn.outputLayer)
nn.inputLayer.computeError(nn.hiddenLayer)
'''






        