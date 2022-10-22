import math

class Neuron:
    def __init__(self, inputData, targetData, weight, activationFunction):
        self.inputData = inputData #[(-1,x1,y1), (-1,x2,y2), etc...]
        self.targetData = targetData #[0, 1, 1, 1, etc...]  (=the real class)
        self.weight = weight #[theta , w1, w2]
        self.activationFunction = activationFunction
        
    def computeSum(self, inputIndex):
        sum = 0
        nbOfWeight = len(self.weight)
        for i in range(nbOfWeight):
            sum += self.inputData[inputIndex][i] * self.weight[i]
        return sum
    
    def applyActFunc(self, sum):
        output = 0
        if self.activationFunction == "heaviside":
            if sum >= 0:
                output = 1
            else:
                output = 0
        if self.activationFunction == "relu":
            if sum >= 0:
                output = sum
            else:
                output = 0
        return output
    
    def computeDerivative(self, sum):
        if self.activationFunction == "heaviside":
            return 1
        if self.activationFunction == "relu":
            if sum >= 0:
                return 1
            else:
                return 0
    
    def computeLoss(self, weight, inputIndex):
        loss = 0
        value = self.inputData[inputIndex][2]
        prediction = (weight[1]/weight[2])* self.inputData[inputIndex][1] + (weight[0]/weight[2])
        loss = math.pow(prediction-value,2)
        return loss
    
    def updateWeight(self, sum, output, target, lr, inputIndex):
        nbOfWeight = len(self.weight)
        derivative = self.computeDerivative(sum)
        for i in range(nbOfWeight):
            delta = lr*(target-output)*derivative*self.inputData[inputIndex][i]
            self.weight[i] = self.weight[i] + delta
            
    def train(self, nbEpoch, lr):
        for index in range(nbEpoch):
            sum = self.computeSum(index)
            output = self.applyActFunc(sum)
            target = self.targetData[index]
            self.updateWeight(sum, output, target, lr, index)
            print("Epoch", index ,"finished ...", " Weight = " ,self.weight, "Loss = ", self.computeLoss(self.weight, index))
        print(self.weight)
        
        
        
        