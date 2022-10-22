import math
class Neuron:
    def __init__(self, lr, actFunc):
        self.weight = [0,0,0]
        self.lr = lr
        self.actFunc = actFunc
        self.inputs = [-1,0, 0] #default value
        self.target = 0 #default value
        self.sum = 0

    def setInput(self, x, y):
        self.inputs = [-1,x, y]

    def setTarget(self, target):
        self.target = target


    def activationFunc(self, sum):
        if self.actFunc == "heaviside":
            if sum >= 0:
                return 1.0
            else:
                return 0.0
        if self.actFunc == "logistic":
            beta = 1
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

    
    def derivativeActFunc(self, sum):
        if self.actFunc == "heaviside":
            return 1.0
        if self.actFunc == "logistic":
            beta = 1.0
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

    #Return the output and the sum
    def prediction(self):
        self.sum = 0
        for i in range(len(self.weight)):
            self.sum += self.inputs[i]*self.weight[i]
        
        return self.activationFunc(self.sum)

    def train(self):
        prediction = self.prediction()
        error = self.target - prediction

        for i in range(len(self.weight)):
            self.weight[i] += error * self.inputs[i] * self.lr * self.derivativeActFunc(self.sum)


    def testPrediction(self):
        prediction = self.prediction()
        self.target
        if(prediction == self.target):
            return "GOOD"
        else:
            return "BAD"
        