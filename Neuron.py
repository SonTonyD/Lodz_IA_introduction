import math

class Neuron:
    def __init__(self, inputData, targetData, outputData, weightVector,activationFunction):
        self.inputData = inputData #[(x1,y1), (x2,y2), etc...]
        self.targetData = targetData #[0, 1, 0, 1, etc...]  (=the real class)
        self.outputData = outputData #[0,1,0,1] for example
        self.weightVector = weightVector
        
        self.activationFunction = activationFunction
    
    def train(self, numberOfEpoch):
        localLoss = 10
        for i in range(numberOfEpoch):
            localLoss = self.targetData - self.inputData
        #Not finished
            
    def computeLoss(self):
        nbValue = self.inputData.size()
        totalLoss = 0
        for i in range(nbValue):
            totalLoss += math.pow(self.outputData[i] - self.targetData[i],2)
        return totalLoss
    
    def computeWeightedInput(self):
        nbValue = self.inputData.size()
        for i in range(nbValue):
            self.inputData[i][0] = self.inputData[i][0] * self.weightVector[i]
        return self.inputData[i][0]

        
            
        
            
        
        
    