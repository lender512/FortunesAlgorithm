import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util import *
from minHeap import *
from parabola import *
from point import *
from searchTree import *
fig, ax = plt.subplots()

step = 0.001
size = 1
n = 2

ax.set_xlim(0, size), ax.set_ylim(0, size)

x = np.sort(np.random.choice(np.arange(0, size, step), size=n))
y = np.random.choice(np.arange(0, size, step), size=n)

#Priorityqueue of points in the plane
Q = MinHeap()

#Insert in priorityQueue
for i in range(n):
    p = Point(x[i], y[i])
    Q.insertKey(p)


parabolasAVL = AVLTree()
root = None
# root = parabolasAVL.insert(None, Parabola(Q.extractMin()))
# root.found = True

plt.scatter(x, y)

#Sweeping line
traversal, = plt.plot((0,0), (0,1))

def intersectionL(A, B, i):
    p = A

    if A.x == B.x:
        y = (A.y + B.y) / 2
    elif A.x == i:
        y = A.y
    elif B.x == i:
        y = B.y
        p = B
    else:
        pA = A.x - i
        pB = B.x - i

        a = (1/(2*pA)) - (1/(2*pB))
        
        b = (-A.y/pA) - (-B.y/pB)

        c = ((A.y*A.y)/(2*pA) + A.x) - ((B.y*B.y)/(2*pB) + B.x) 

        y = (-b - np.sqrt(b*b - 4*a*c))/ (2*a)

    x = (p.x*p.x + (p.y-y)*(p.y-y) - i*i) / (2*p.x - 2*i)
    # x =(p.x*p.x + (p.y-y)*(p.y-y))/(2*(p.x-i))
    # x = a*y*y + b*y +c

    return x, y

def intersectionU(A, B, i):
    p = A

    if A.x == B.x:
        y = (A.y + B.y) / 2
    elif A.x == i:
        y = A.y
    elif B.x == i:
        y = B.y
        p = B
    else:
        pA = A.x - i
        pB = B.x - i

        a = (1/(2*pA)) - (1/(2*pB))
        
        b = (-A.y/pA) - (-B.y/pB)

        c = ((A.y*A.y)/(2*pA) + A.x) - ((B.y*B.y)/(2*pB) + B.x) 

        y = (-b + np.sqrt(b*b - 4*a*c))/ (2*a)

    x = (p.x*p.x + (p.y-y)*(p.y-y) - i*i) / (2*p.x - 2*i)
    # x =(p.x*p.x + (p.y-y)*(p.y-y))/(2*(p.x-i))
    # x = a*y*y + b*y +c

    return x, y


def update(parabola, i):
        xCoord = parabola.vertex.x
        yCoord = parabola.vertex.y

        parXU = np.arange(parabola.upperLimit[0], parabola.mostRight, step/40)
        p = i-xCoord
        parYU = np.sqrt(-(2*p*(parXU-i+p/2)))+yCoord
        parabola.upperLimit = parabola.upperLimit[0], np.sqrt(-(2*p*(parabola.upperLimit[0]-i+p/2)))+yCoord
        
        parabola.parUpper.set_xdata(parXU)
        parabola.parUpper.set_ydata(parYU)

        parXL = np.arange(parabola.lowerLimit[0], parabola.mostRight, step/40)
        parYL = -np.sqrt(-(2*p*(parXL-i+p/2)))+yCoord
        parabola.lowerLimit = parabola.lowerLimit[0], -np.sqrt(-(2*p*(parabola.lowerLimit[0]-i+p/2)))+yCoord

        parabola.parLower.set_xdata(parXL)
        parabola.parLower.set_ydata(parYL)

        upperParabola = parabolasAVL.search(root,yCoord)

        if upperParabola is not None and upperParabola.value.vertex != parabola.vertex:
            parabola.found = True
            upperPoint = intersectionU(upperParabola.value.vertex, parabola.vertex, i)
            lowerPoint = intersectionL(upperParabola.value.vertex, parabola.vertex, i)
            parabola.upperLimit = upperPoint
            parabola.lowerLimit = lowerPoint
            # upperParabola.value.lowerLimit = point
            # plt.scatter(point[0], point[1])

                # xU = upperParabola.value.parUpper.get_data()[0]
                # fU = upperParabola.value.parUpper.get_data()[1]
                # gU = parYU
                # idx = np.argwhere(np.diff(np.sign(fU - gU))).flatten()
                # plt.plot(xU[idx], fU[idx], 'ro')

                # xL = upperParabola.value.parLower.get_data()[0]
                # fL = upperParabola.value.parLower.get_data()[1]
                # gL = parYL
                # idx = np.argwhere(np.diff(np.sign(fL - gL))).flatten()
                # plt.plot(parXL[idx], fL[idx], 'ro')
                # # print([value for value in upperParabola.value.parY if value in parabola.parY])
                # # current = Q.extractMin()


        if i >= size-0.01:
            parabola.parUpper.set_xdata([])
            parabola.parUpper.set_ydata([])
            parabola.parLower.set_xdata([])
            parabola.parLower.set_ydata([])


def inOrderUpdate(root, i):
    if root is not None:
        inOrderUpdate(root.l, i)
        update(root.value, i)
        inOrderUpdate(root.r, i)


def animationFrame(i):
    traversal.set_xdata(i)

    global root
    if (len(Q.heap) > 0 and Q.getMin().x < i):
        if root is None:
            root = parabolasAVL.insert(root, Parabola(Q.extractMin()))
        else:
            parabolasAVL.insert(root, Parabola(Q.extractMin()))

    inOrderUpdate(root, i)


    return traversal
    

animation = FuncAnimation(fig, func=animationFrame, frames=np.arange(0, size, step), interval=1)


plt.show()