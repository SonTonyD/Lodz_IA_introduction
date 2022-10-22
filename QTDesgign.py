from PyQt5 import QtWidgets, uic
import sys
import pyqtgraph as pg
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


        
        print(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass)
        
        plot = pg.plot()
        layout = QGridLayout()
        
        self.plotComponent(plot, layout, meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass)
    
    def plotComponent(self, plot, layout, meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass):
        scatter01 = self.generate_points(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, 'y')
        scatter02 = self.generate_points(meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, 'b')
        
        plot.addItem(scatter01)
        plot.addItem(scatter02)
        
        #Create input numpy.array and label each data
        inputClass1 = np.array([scatter01.getData()[0][0],scatter01.getData()[1][0],0])
        for i in range(1,len(scatter01.getData()[0])):
            inputClass1 = np.vstack([inputClass1, [scatter01.getData()[0][i],scatter01.getData()[1][i],0]])
            
        inputClass2 = np.array([scatter02.getData()[0][0],scatter02.getData()[1][0],1])
        for i in range(1,len(scatter02.getData()[0])):
            inputClass2 = np.vstack([inputClass2, [scatter02.getData()[0][i],scatter02.getData()[1][i],1]])  
        
        
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
        #print(inputData)


        nbEpoch = 200
        lr = 0.05
        n = Neuron(lr)

        #Test Precision
        precision = 0
        for i in range(nbEpoch):
            n.setInput(inputData[i][0], inputData[i][1])
            n.setTarget(target[i])
            if(n.testPrediction() == "GOOD"):
                precision += 1 
        precision = (precision/nbEpoch)*100
        print("Precision before training: ", precision, " %")

        for i in range(nbEpoch):
            n.setInput(inputData[i][0], inputData[i][1])
            n.setTarget(target[i])
            n.train()

        #Test Precision
        testSet=50
        precision = 0
        for i in range(nbEpoch, nbEpoch+testSet):
            n.setInput(inputData[i][0], inputData[i][1])
            n.setTarget(target[i])
            if(n.testPrediction() == "GOOD"):
                precision += 1 
        precision = (precision/testSet)*100
        print("Precision after training on TestSet: ", precision, " %")

        '''
        weight = [0,1,1]
        neuron = Neuron(inputData, target, weight, "relu")
        lr = 0.06
        neuron.train(180,lr)
        '''
        x = np.arange(meanMin, meanMax, 0.1)
        y = np.arange(meanMin, meanMax, 0.1)

        Z = np.array([])
        X, Y = np.meshgrid(x,y)
        for i in x:
            for j in y:
                n.setInput(i,j)
                Z = np.append(Z, n.prediction())
        Z = Z.reshape((len(x), len(y)))
        plt.contourf(x,y,Z,cmap="bone")
        plt.show()
        
                
                







        
        layout.addWidget(plot)
        #self.widget.setLayout(layout)
        
    def generate_points(self, meanMin, meanMax, varMin, varMax, samplePerMode, modePerClass, color):
        scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(color))
        
        for i in range(modePerClass):
            x = (meanMax - meanMin)*np.random.random_sample() + meanMin
            y = (meanMax - meanMin)*np.random.random_sample() + meanMin
            
            sigma = (varMax - varMin)*np.random.random_sample() + varMin
            
            posX = np.random.normal(x, sigma, samplePerMode)
            posY = np.random.normal(y, sigma, samplePerMode)
            scatter.addPoints(posX, posY)
        return scatter
        
app = QtWidgets.QApplication(sys.argv)
window = DemoWidget()
app.exec_()