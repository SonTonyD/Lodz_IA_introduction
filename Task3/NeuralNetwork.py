import math
import numpy as np
from Layer import Layer
class NeuralNetwork:
    def __init__(self, input, nbInputNeuron, nbHiddenNeuron, nbOutputNeuron):
        self.input = input
        self.inputLayer = Layer('relu', nbInputNeuron, 0.1)
        self.hiddenLayer = Layer('relu', nbHiddenNeuron, 0.1)
        self.outputLayer = Layer('relu', nbOutputNeuron, 0.1)

        self.Layers = [self.inputLayer, self.hiddenLayer, self.outputLayer]
    
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



input = np.array([3,2])
target = np.array([0,1])


nn = NeuralNetwork(input, 2, 3, 2)

nn.initAllWeights()
#nn.feedforward()



print(nn.outputLayer.output)
        