import wx
import math
import matplotlib.pyplot as plt
import numpy as np
import random as rdm

class MyFrame(wx.Frame):
        
    def __init__(self):
        super().__init__(parent=None, title='Hello World')
        panel = wx.Panel(self)

        self.label_pointsX = wx.StaticText(panel ,label="Position X", pos=(6, 10))
        self.textX = wx.TextCtrl(panel, pos=(5, 30))
        
        self.label_pointsY = wx.StaticText(panel ,label="Position Y", pos=(6, 10+50))
        self.textY = wx.TextCtrl(panel, pos=(5, 30+50))
        
        self.label_pointsY = wx.StaticText(panel ,label="Number of points", pos=(6, 10+50+50))
        self.textNbPoints = wx.TextCtrl(panel, pos=(5, 30+50+50))
        
        self.submitBtn = wx.Button(panel, label='Generate Points', pos=(5, 160))
        self.submitBtn.Bind(wx.EVT_BUTTON, self.OnClicked)
        
        self.Show()
    
    def OnClicked(self, event): 
        btn = event.GetEventObject().GetLabel()
        
        posX = int(self.textX.GetValue())
        posY = int(self.textY.GetValue())
        nbPoints = int(self.textNbPoints.GetValue())
        
        self.define_class(posX, posY, 1, nbPoints, 'o')
        
        plt.xlim(-2, 15)
        plt.ylim(-2, 15)

        plt.show()
        
        print ("Label of pressed button = ",btn)
        
    def define_class(e ,posX, posY, radius, nbPoints, marker):
        xpoints = np.array([])
        ypoints = np.array([])
        
        for i in range(nbPoints):
            r = rdm.random()*radius
            theta = rdm.randint(0,360)
            xpoints = np.insert(xpoints, 0, r*math.cos(theta) + posX)
            ypoints = np.insert(ypoints, 0, r*math.sin(theta) + posY)
        
        plt.plot(xpoints, ypoints, marker)

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
    