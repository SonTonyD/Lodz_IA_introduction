class Neuron:
    def __init__(self, lr):
        self.weight = [1,-0.5,0.5]
        self.lr = lr
        self.inputs = [-1, 0, 0] #default value
        self.target = 0 #default value

    def setInput(self, x, y):
        self.inputs = [-1, x, y]

    def setTarget(self, target):
        self.target = target


    def heaviside(self, sum):
        if sum >= 0:
            return 1
        else:
            return 0
        
    def prediction(self):
        sum = 0
        for i in range(len(self.weight)):
            sum += self.inputs[i]*self.weight[i]
        return self.heaviside(sum)

    def train(self):
        prediction = self.prediction()
        error = self.target - prediction

        for i in range(len(self.weight)):
            self.weight[i] += error * self.inputs[i] * self.lr

    def testPrediction(self):
        prediction = self.prediction()
        self.target
        if(prediction == self.target):
            return "GOOD"
        else:
            return "BAD"
        