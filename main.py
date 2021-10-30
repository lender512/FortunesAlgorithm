import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util import *
from minHeap import *
from parabola import *
from point import *

fig, ax = plt.subplots()

step = 0.001
size = 1
n = 6

ax.set_xlim(0, size), ax.set_ylim(0, size)

x = np.random.choice(np.arange(0, size, step), size=n)
y = np.random.choice(np.arange(0, size, step), size=n)

#Priorityqueue of points in the plane
Q = MinHeap()

#Insert in priorityQueue
for i in range(n):
    p = Point(x[i], y[i])
    Q.insertKey(p)


parabolas = []

for i in range(n):
    parabolas.append(Parabola())

plt.scatter(x, y)

#Sweeping line
traversal, = plt.plot((0,0), (0,1))

def animationFrame(i):
    traversal.set_xdata(i)

    
    for j in range(n):
        xCoord = x[j]
        yCoord = y[j]

        if xCoord < i:
            
            parX = np.arange(parabolas[j].upperLimit, i, step/40)
            p = i-xCoord
            parY = np.sqrt(-(2*p*(parX-i+p/2)))+yCoord

            
            parabolas[j].parUpper.set_xdata(parX)
            parabolas[j].parUpper.set_ydata(parY)

            parX = np.arange(parabolas[j].lowerLimit, i, step/40)
            parY = -np.sqrt(-(2*p*(parX-i+p/2)))+yCoord

            parabolas[j].parLower.set_xdata(parX)
            parabolas[j].parLower.set_ydata(parY)

            if not parabolas[j].found:
                parabolas[j].found = True
                print("linearEvent")

                # current = Q.extractMin()


        if i >= size-0.01:
            parabolas[j].parUpper.set_xdata([])
            parabolas[j].parUpper.set_ydata([])
            parabolas[j].parLower.set_xdata([])
            parabolas[j].parLower.set_ydata([])


    return traversal
    

animation = FuncAnimation(fig, func=animationFrame, frames=np.arange(0, size, step), interval=1)


plt.show()