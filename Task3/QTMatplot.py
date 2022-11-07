import math
from PyQt5 import QtWidgets, uic
import sys
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Neuron_3 import Neuron
import matplotlib.pyplot as plt
from matplotlib import colors
from PyQt5.QtCore import *


class DemoWidget(QtWidgets.QWidget):
    def __init__(self):
        super(DemoWidget, self).__init__()
        uic.loadUi('gui_plot.ui', self)
        self.label_betaValue.setHidden(True)
        self.lineEdit_betaValue.setHidden(True)
        
        self.pushButton.clicked.connect(self.click)
        self.defaultValueButton.clicked.connect(self.setDefaultValue)
        self.comboBoxActFunc.currentIndexChanged.connect(self.selectionChange)
        self.checkBox_variableLr.stateChanged.connect(self.handleCheckBox)

        self.show()

    def handleCheckBox(self):
        if self.checkBox_variableLr.isChecked() == True:
            self.lineEdit_minLr.setEnabled(True)
            self.lineEdit_maxLr.setEnabled(True)
            self.lineEdit_lr.setEnabled(False)
        else:
            self.lineEdit_minLr.setEnabled(False)
            self.lineEdit_maxLr.setEnabled(False)
            self.lineEdit_lr.setEnabled(True)

    def selectionChange(self):
        if self.comboBoxActFunc.currentText() == "logistic":
            self.label_betaValue.setHidden(False)
            self.lineEdit_betaValue.setHidden(False)
        else:
            self.label_betaValue.setHidden(True)
            self.lineEdit_betaValue.setHidden(True)

    def setDefaultValue(self):
        plt.close()
        self.lineEdit_meanMin.setText("0")
        self.lineEdit_meanMax.setText("10")
        
        self.lineEdit_varMin.setText("0.5")
        self.lineEdit_varMax.setText("0.9")
        
        self.lineEdit_samplePerMode.setText("300")
        self.lineEdit_modePerClass.setText("1")

        self.lineEdit_lr.setText("0.01")
        self.lineEdit_nbOfEpoch.setText("200")

        self.lineEdit_betaValue.setText("1.0")

        self.lineEdit_nbNeuronsInput.setText("2")
        self.lineEdit_nbNeuronsHidden.setText("3")


    def click(self):
        plt.close()
        meanMin = float(self.lineEdit_meanMin.text())
        meanMax = float(self.lineEdit_meanMax.text())
        
        varMin = float(self.lineEdit_varMin.text())
        varMax = float(self.lineEdit_varMax.text())
        
        samplePerMode = int(self.lineEdit_samplePerMode.text())
        modePerClass = int(self.lineEdit_modePerClass.text())

        if self.checkBox_variableLr.isChecked() == True:
            lr = float(self.lineEdit_minLr.text())
            maxLr = float(self.lineEdit_maxLr.text())
        else:
            lr = float(self.lineEdit_lr.text())
            maxLr = float(self.lineEdit_lr.text())


        nbOfEpoch = int(self.lineEdit_nbOfEpoch.text())
        actFunc = self.comboBoxActFunc.currentText()
        betaValue = float(self.lineEdit_betaValue.text())

        nbInputNeuron = int(self.lineEdit_nbNeuronsInput.text())
        nbHiddenNeuron = int(self.lineEdit_nbNeuronsHidden.text())

        self.plotComponent(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, lr, nbOfEpoch, actFunc, betaValue, maxLr, nbInputNeuron, nbHiddenNeuron)

    def plotComponent(self, meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, lr, nbOfEpoch, actFunc, betaValue, maxLr, nbInputNeuron, nbHiddenNeuron):
        arrayClass01X = np.array([])
        arrayClass01Y = np.array([])

        arrayClass02X = np.array([])
        arrayClass02Y = np.array([])

        (arrayClass01X, arrayClass01Y) = self.generate_points(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, "blue")
        (arrayClass02X, arrayClass02Y) = self.generate_points(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, "red")

        #Create inputData and target as numpy arrays
        inputData, target = self.formatData(arrayClass01X, arrayClass01Y, arrayClass02X, arrayClass02Y)

        #Feed the neuron
        #neuron = Neuron(lr, actFunc, betaValue)
        #neuron.weight = self.useNeuron(inputData, target, nbOfEpoch, lr, actFunc, betaValue, maxLr)

        #Draw Contour
        #self.drawContour(neuron, meanMin, meanMax)

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

        #build target vector from AllInput vector
        target = np.array([])
        for i in range(allInput.shape[0]):
            target = np.append(target, allInput[i][2])
        #print(target)
        
        #re-build allInput without target
        inputData = np.array([allInput[0][0],allInput[0][1]])
        for i in range(1,allInput.shape[0]):
            inputData = np.vstack([inputData, [allInput[i][0], allInput[i][1]] ])
        #print(inputData)
        
        return inputData, target

    def useNeuron(self, inputData, targetData, nbEpoch, lr, actFunc, betaValue, maxLr):
        n = Neuron(lr, actFunc, betaValue)

        
        n.initNeuronWeight(inputData[0])
        for i in range(nbEpoch):
            n.lr = lr + (maxLr-lr)*(1+(math.cos((i*math.pi)/nbEpoch)))
            n.setInput(inputData[i%(inputData.shape[0])])
            n.setTarget(targetData[i%(inputData.shape[0])])

            #print(inputData[i%(inputData.shape[0])][0], inputData[i%(inputData.shape[0])][1], targetData[i%(inputData.shape[0])], n.prediction())
            n.train()
        
        
        return n.weight


    def drawContour(self, neuron, meanMin, meanMax):
        x = np.arange(meanMin, meanMax, 0.1)
        y = np.arange(meanMin, meanMax, 0.1)

        Z = np.array([])
        X, Y = np.meshgrid(x,y)
        for i in x:
            for j in y:

                neuron.setInput((j,i))
                Z = np.append(Z, neuron.prediction())
                #print(i,j,neuron.prediction())
        Z = Z.reshape((len(x), len(y)))


        cs = plt.contourf(X,Y,Z,cmap="jet",alpha = 0.2)
        cbar = plt.colorbar(cs)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DemoWidget()
    app.exec_()
