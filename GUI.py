import wx
import math
import matplotlib.pyplot as plt
import numpy as np
import random as rdm

class MyFrame(wx.Frame):
        
    def __init__(self):
        super().__init__(parent=None, title='Artificial Intelligence Plotting', size=(400, 600))
        panel = wx.Panel(self)

        #Class Settings
        base_pos_y = 30
        self.label_ClassSettings = wx.StaticText(panel ,label="Class Settings", pos=(10, base_pos_y-20))
        
        self.label_pointsX = wx.StaticText(panel ,label="Position X", pos=(6, base_pos_y))
        self.textX = wx.TextCtrl(panel, pos=(5, base_pos_y+20))
        
        self.label_pointsY = wx.StaticText(panel ,label="Position Y", pos=(6, base_pos_y+50))
        self.textY = wx.TextCtrl(panel, pos=(5, base_pos_y+20+50))
        
        self.label_radius = wx.StaticText(panel ,label="Max Radius", pos=(6, base_pos_y+50+50))
        self.textRadius = wx.TextCtrl(panel, pos=(5, base_pos_y+20+50+50))
        
        self.label_pointsY = wx.StaticText(panel ,label="Number of points", pos=(6, base_pos_y+50+50+50))
        self.textNbPoints = wx.TextCtrl(panel, pos=(5, base_pos_y+20+50+50+50))
        
        
        
        #Plot Settings
        base_pos_y = 260
        self.label_ClassSettings = wx.StaticText(panel ,label="Plot Settings", pos=(10, base_pos_y-20))
        
        self.label_pointsX = wx.StaticText(panel ,label="minX", pos=(6, base_pos_y))
        self.textMinX = wx.TextCtrl(panel, pos=(5, base_pos_y+20))
        
        self.label_pointsY = wx.StaticText(panel ,label="minY", pos=(6, base_pos_y+50))
        self.textMinY = wx.TextCtrl(panel, pos=(5, base_pos_y+20+50))
        
        self.label_radius = wx.StaticText(panel ,label="maxX", pos=(130, base_pos_y))
        self.textMaxX = wx.TextCtrl(panel, pos=(130, base_pos_y+20))
        
        self.label_pointsY = wx.StaticText(panel ,label="maxY", pos=(130, base_pos_y+50))
        self.textMaxY = wx.TextCtrl(panel, pos=(130, base_pos_y+20+50))
        
        #Submit Button
        self.submitBtn = wx.Button(panel, label='Generate Points', pos=(130, 30))
        self.submitBtn.Bind(wx.EVT_BUTTON, self.OnClicked)
        
        #Default Button
        self.setDefaultPlot = wx.Button(panel, label='Set Default Plot', pos=(130, 260-25))
        self.setDefaultPlot.Bind(wx.EVT_BUTTON, self.SetDefaultPlot)
        
        self.Show()
    
    def OnClicked(self, event): 
        btn = event.GetEventObject().GetLabel()
        
        posX, posY = (float(self.textX.GetValue()), float(self.textY.GetValue()))
        nbPoints = int(self.textNbPoints.GetValue())
        radius = float(self.textRadius.GetValue())
        
        self.define_class(posX, posY, radius, nbPoints, 'o')
        
        
        if self.textMinX.GetValue() != None:
            minX, maxX, minY, maxY = (float(self.textMinX.GetValue()), float(self.textMaxX.GetValue()), float(self.textMinY.GetValue()), float(self.textMaxY.GetValue()))
        else:
            minX, maxX, minY, maxY = (-2, 12, -2, 12)
        
        plt.xlim(minX, maxX)
        plt.ylim(minY, maxY)

        plt.show()
        
        print ("Label of pressed button = ",btn)
        
    def SetDefaultPlot(self, event):
        print("Set Default Value")
        self.textMinX.SetValue("-2")
        self.textMaxX.SetValue("12")
        self.textMinY.SetValue("-2")
        self.textMaxY.SetValue("12")
        
        
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
    