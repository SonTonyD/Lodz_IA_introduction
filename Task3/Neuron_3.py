import math
import numpy as np
class Neuron:
    def __init__(self, lr, actFunc, betaValue):
        self.weight = np.zeros(0)
        self.lr = lr
        self.actFunc = actFunc
        self.inputs = [] #default value
        self.target = 0 #default value
        self.sum = 0
        self.betaValue = betaValue
        self.error = 0

    def initNeuronWeight(self, inputs):
        self.weight = np.random.rand(inputs.shape[0] + 1)

    def setInput(self, inputs):
        self.inputs = inputs
        self.inputs = np.insert(self.inputs, 0, -1)

    def setTarget(self, target):
        self.target = target


    def activationFunc(self, sum):
        if self.actFunc == "heaviside":
            if sum >= 0:
                return 1.0
            else:
                return 0.0
        if self.actFunc == "logistic":
            beta = self.betaValue
            if -beta*sum > 450:
                return 1/(1+math.exp(450))
            return 1/(1+math.exp(-beta*sum))
        if self.actFunc == "sign":
            if sum >= 0:
                return 1.0
            else:
                return -1.0
        if self.actFunc == "relu":
            return max(0,sum)
        if self.actFunc == "sin":
            return math.sin(sum)
        if self.actFunc == "tanh":
            return math.tanh(sum)
        if self.actFunc == "leaky relu":
            if sum > 0:
                return sum
            else :
                return 0.01*sum

    
    def derivativeActFunc(self, sum):
        if self.actFunc == "heaviside":
            return 1.0
        if self.actFunc == "logistic":
            beta = self.betaValue
            if -beta*sum > 300:
                phi = 1/(1+math.exp(300))
            else:
                phi = 1/(1+math.exp(-beta*sum))

            return beta*phi*(1-phi)
        if self.actFunc == "sign":
            return 1.0
        if self.actFunc == "relu":
            if sum >= 0:
                return 1.0
            else:
                return 0.0
        if self.actFunc == "sin":
            return math.cos(sum)
        if self.actFunc == "tanh":
            return math.pow(1/(math.cosh(sum)),2)
        if self.actFunc == "leaky relu":
            if sum > 0:
                return 1
            else :
                return 0.01

    #Return the output y
    def prediction(self):
        self.sum = 0
        for i in range(len(self.inputs)):
            self.sum += self.inputs[i]*self.weight[i]

        
        return self.activationFunc(self.sum)

    def train(self):
        prediction = self.prediction()
        error = self.target - prediction

        for i in range(len(self.weight)):
            self.weight[i] += error * self.inputs[i] * self.lr * self.derivativeActFunc(self.sum)

    def computeErrorOutput(self):
        prediction = self.prediction()
        self.error = self.derivativeActFunc(self.sum)*(self.target - prediction)
        return self.error


    def computeError(self, previousErrors, previousWeights):
        prediction = self.prediction() #keep this, to compute self.sum !!

        weightedErrors = 0
        for i in range(len(previousErrors)):
            weightedErrors += previousWeights[i]*previousErrors[i] 

        self.error = self.derivativeActFunc(self.sum)*weightedErrors
        return self.error
    
    def updateWeights(self):
        for i in range(1,len(self.weight)):
            self.weight[i] += self.lr * self.inputs[i] * self.error





        