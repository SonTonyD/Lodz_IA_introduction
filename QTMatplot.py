from PyQt5 import QtWidgets, uic
import sys
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Neuron_3 import Neuron
import matplotlib.pyplot as plt

class DemoWidget(QtWidgets.QWidget):
    def __init__(self):
        super(DemoWidget, self).__init__()
        uic.loadUi('gui_plot.ui', self)
        
        self.pushButton.clicked.connect(self.click)
        self.show()

    def click(self):
        meanMin = float(self.lineEdit_meanMin.text())
        meanMax = float(self.lineEdit_meanMax.text())
        
        varMin = float(self.lineEdit_varMin.text())
        varMax = float(self.lineEdit_varMax.text())
        
        samplePerMode = int(self.lineEdit_samplePerMode.text())
        modePerClass = int(self.lineEdit_modePerClass.text())

        self.plotComponent(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass)

    def plotComponent(self, meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass):
        arrayClass01X = np.array([])
        arrayClass01Y = np.array([])

        arrayClass02X = np.array([])
        arrayClass02Y = np.array([])

        (arrayClass01X, arrayClass01Y) = self.generate_points(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, "red")
        (arrayClass02X, arrayClass02Y) = self.generate_points(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, "blue")

        #Create input numpy.array and label each data
        inputData, target = self.formatData(arrayClass01X, arrayClass01Y, arrayClass02X, arrayClass02Y)

        #Feed the neuron
        nbEpoch = 200
        lr = 0.05
        neuron = self.useNeuron(inputData, target, nbEpoch, lr)

        #Draw Contour
        self.drawContour(neuron, meanMin, meanMax)

        plt.xlim(meanMin, meanMax)
        plt.ylim(meanMin, meanMax)
        plt.show()

    def generate_points(self, meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, color):
        arrayX = np.array([])
        arrayY = np.array([])
        for i in range(modePerClass):
            x = (meanMax - meanMin)*np.random.random_sample() + meanMin
            y = (meanMax - meanMin)*np.random.random_sample() + meanMin
            
            sigma = (varMax - varMin)*np.random.random_sample() + varMin
            
            posX = np.random.normal(x, sigma, samplePerMode)
            posY = np.random.normal(y, sigma, samplePerMode)

            arrayX = np.append(arrayX, posX)
            arrayY = np.append(arrayY, posY)
        plt.scatter(arrayX, arrayY, c = color)
        return arrayX, arrayY
    
    def formatData(self, arrayClass01X, arrayClass01Y, arrayClass02X, arrayClass02Y):
        inputClass1 = np.array([arrayClass01X[0],arrayClass01Y[0],0])
        for i in range(1,len(arrayClass01X)):
            inputClass1 = np.vstack([inputClass1, [arrayClass01X[i],arrayClass01Y[i],0]])
            
        inputClass2 = np.array([arrayClass02X[0],arrayClass02Y[0],1])
        for i in range(1,len(arrayClass02X)):
            inputClass2 = np.vstack([inputClass2, [arrayClass02X[i],arrayClass02Y[i],1]])
        
        #Concatenate all data and shuffle it
        allInput = np.vstack([inputClass1,inputClass2])
        np.random.shuffle(allInput)
        #print(allInput)

        #build target vector
        target = np.array([])
        for i in range(allInput.shape[0]):
            target = np.append(target, allInput[i][2])
        #print(target)
        
        #re-build allInput without target
        inputData = np.array([allInput[0][0],allInput[0][1]])
        for i in range(1,allInput.shape[0]):
            inputData = np.vstack([inputData, [allInput[i][0], allInput[i][1]] ])
        
        return inputData, target

    def useNeuron(self, inputData, targetData, nbEpoch, lr):
        n = Neuron(lr)

        for i in range(nbEpoch):
            n.setInput(inputData[i][0], inputData[i][1])
            n.setTarget(targetData[i])
            n.train()
        
        return n

    def drawContour(self, neuron, meanMin, meanMax):
        x = np.arange(meanMin, meanMax, 0.1)
        y = np.arange(meanMin, meanMax, 0.1)

        Z = np.array([])
        X, Y = np.meshgrid(x,y)
        for i in x:
            for j in y:
                neuron.setInput(i,j)
                Z = np.append(Z, neuron.prediction())
        Z = Z.reshape((len(x), len(y)))
        plt.contourf(x,y,Z,cmap="bone",alpha = 0.5)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DemoWidget()
    app.exec_()
