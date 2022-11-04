import math
class NeuralNetwork:
    def __init__(self, data, nbInput, nbHidden, nbOutput):
        self.data = data
        self.nbInput = nbInput
        self.nbHidden = nbHidden
        self.nbOutput = nbOutput
        