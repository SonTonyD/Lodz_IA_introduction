import math
from re import X
import matplotlib.pyplot as plt
import numpy as np
import random as rdm

'''
#Initialisation Value
xpoints1 = np.array([])
ypoints1 = np.array([])

xpoints2 = np.array([])
ypoints2 = np.array([])

xpoints3 = np.array([])
ypoints3 = np.array([])

#Mode Postion mode1 = (x, y, spread)
mode1 = (2,2,2)
mode2 = (5,5,2)
mode3 = (3,4,2)


for i in range(50):
    #Mode 1 
    xpoints1 = np.insert(xpoints1, 0, rdm.random()*mode1[2] + mode1[0])
    ypoints1 = np.insert(ypoints1, 0, rdm.random()*mode1[2] + mode1[1])
    
    #Mode 2 
    xpoints2 = np.insert(xpoints2, 0, rdm.random()*mode2[2] + mode2[0])
    ypoints2 = np.insert(ypoints2, 0, rdm.random()*mode2[2] + mode2[1])
    
    #Mode 3
    xpoints3 = np.insert(xpoints3, 0, rdm.random()*mode3[2] + mode3[0])
    ypoints3 = np.insert(ypoints3, 0, rdm.random()*mode3[2] + mode3[1])





plt.plot(xpoints1, ypoints1, 'o')
plt.plot(xpoints2, ypoints2, 'v')
plt.plot(xpoints3, ypoints3, 'x')
plt.show()
'''

'''
def define_class(posX, posY, radius, nbPoints, marker):
    xpoints = np.array([])
    ypoints = np.array([])
    
    for i in range(nbPoints):
        r = rdm.random()*radius
        theta = rdm.randint(0,360)
        xpoints = np.insert(xpoints, 0, r*math.cos(theta) + posX)
        ypoints = np.insert(ypoints, 0, r*math.sin(theta) + posY)
    
    plt.plot(xpoints, ypoints, marker)


define_class(2, 5, 2, 200, 'o')
define_class(7, 3, 2, 200, 'o')

plt.xlim(-2, 15)
plt.ylim(-2, 15)

plt.show()

'''



'''
minMean
maxMean
minVar
maxVar
SamplePerMode
ModePerClass
'''

def generatePoints(minMean, maxMean, minVar, maxVar, SamplePerMode, ModePerClass):
    pass



    