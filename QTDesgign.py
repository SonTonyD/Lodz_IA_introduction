from PyQt5 import QtWidgets, uic
import sys
import pyqtgraph as pg
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

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
        
        
        layout.addWidget(plot)
        self.widget.setLayout(layout)
        
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