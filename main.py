import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util import *
from minHeap import *
from parabola import *
from point import *
import math
fig, ax = plt.subplots()

step = 0.001
size = 1
n = 3

ax.set_xlim(0, size), ax.set_ylim(0, size)

x = np.sort(np.random.choice(np.arange(0, size, step), size=n))
y = np.random.choice(np.arange(0, size, step), size=n)

#Priorityqueue of points in the plane
Q = MinHeap()

#Insert in priorityQueue
for i in range(n):
    p = Point(x[i], y[i])
    Q.insertKey(p)

root = None
lastAdded = None
# root = parabolasAVL.insert(None, Parabola(Q.extractMin()))
# root.found = True

plt.scatter(x, y)

#Sweeping line
traversal, = plt.plot((0,0), (0,1))

def intersection(p0, p1, l):
    # get the intersection of two parabolas
    p = p0
    if (p0.x == p1.x):
        py = (p0.y + p1.y) / 2.0
    elif (p1.x == l):
        py = p1.y
    elif (p0.x == l):
        py = p0.y
        p = p1
    else:
        # use quadratic formula
        z0 = 2.0 * (p0.x - l)
        z1 = 2.0 * (p1.x - l)

        a = 1.0/z0 - 1.0/z1;
        b = -2.0 * (p0.y/z0 - p1.y/z1)
        c = 1.0 * (p0.y**2 + p0.x**2 - l**2) / z0 - 1.0 * (p1.y**2 + p1.x**2 - l**2) / z1

        py = 1.0 * (-b-math.sqrt(b*b - 4*a*c)) / (2*a)

    px = 1.0 * (p.x**2 + (p.y-py)**2 - l**2) / (2*p.x-2*l)
    res = Point(px, py)
    return res

def intersectionL(p0, p1, l):
    # get the intersection of two parabolas
    p = p0
    if (p0.x == p1.x):
        py = (p0.y + p1.y) / 2.0
    elif (p1.x == l):
        py = p1.y
    elif (p0.x == l):
        py = p0.y
        p = p1
    else:
        # use quadratic formula
        z0 = 2.0 * (p0.x - l)
        z1 = 2.0 * (p1.x - l)

        a = 1.0/z0 - 1.0/z1;
        b = -2.0 * (p0.y/z0 - p1.y/z1)
        c = 1.0 * (p0.y**2 + p0.x**2 - l**2) / z0 - 1.0 * (p1.y**2 + p1.x**2 - l**2) / z1

        py = 1.0 * (-b+math.sqrt(b*b - 4*a*c)) / (2*a)

    px = 1.0 * (p.x**2 + (p.y-py)**2 - l**2) / (2*p.x-2*l)
    res = Point(px, py)
    return res

def update(parabola, i):
        xCoord = parabola.vertex.x
        yCoord = parabola.vertex.y




        if  parabola.next is not None:
            if parabola.vertexU is None:
                inter = intersection(parabola.vertex, parabola.next.vertex, i)
            else:
                inter = intersection(parabola.vertex, parabola.vertexU, i)
            if not parabola.foundU:
                parabola.foundU = True
                parabola.initU = inter.x, inter.y
                parabola.vertexU =  parabola.next.vertex
            if not parabola.endU:
                if parabola.foundL:
                    parabola.upperSegment.set_data([inter.x, parabola.initL[0]], [inter.y, parabola.initL[1]])
                else:
                    parabola.upperSegment.set_data([inter.x, parabola.initU[0]], [inter.y, parabola.initU[1]])

            parabola.upperLimit = parabola.next.lowerLimit= inter
            # parabola.next.lowerLimit = inter

        parXU = np.arange(parabola.upperLimit.x, i, step/40)
        p = i-xCoord
        parYU = np.sqrt(-(2*p*(parXU-i+p/2)))+yCoord

        parabola.parUpper.set_xdata(parXU)
        parabola.parUpper.set_ydata(parYU)

        if  parabola.prev is not None :
            if parabola.vertexL is None:
                inter = intersectionL(parabola.vertex, parabola.prev.vertex, i)
            else: 
                inter = intersectionL(parabola.vertex, parabola.prev.vertex, i)
                
            if not parabola.foundL:
                parabola.foundL = True
                parabola.initL = inter.x, inter.y
                parabola.vertexL =  parabola.prev.vertex
            if not parabola.endL: 
                if parabola.foundU:
                    parabola.lowerSegment.set_data([inter.x, parabola.initU[0]], [inter.y, parabola.initU[1]])
                else:
                    parabola.lowerSegment.set_data([inter.x, parabola.initL[0]], [inter.y, parabola.initL[1]])
                    
            
            parabola.lowerLimit = parabola.prev.upperLimit  = inter
            # parabola.prev.upperLimit  = inter

        parXL = np.arange(parabola.lowerLimit.x, i, step/40)
        parYL = -np.sqrt(-(2*p*(parXL-i+p/2)))+yCoord

        parabola.parLower.set_xdata(parXL)
        parabola.parLower.set_ydata(parYL)


        # if upperParabola is not None and upperParabola.value.vertex != parabola.vertex:
        #     parabola.found = True
        #     upperPoint = intersectionU(upperParabola.value.vertex, parabola.vertex, i)
        #     lowerPoint = intersectionL(upperParabola.value.vertex, parabola.vertex, i)
        #     parabola.upperLimit = upperPoint
        #     parabola.lowerLimit = lowerPoint
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


        if i >= size*1.5-0.01:
            parabola.parUpper.set_xdata([])
            parabola.parUpper.set_ydata([])
            parabola.parLower.set_xdata([])
            parabola.parLower.set_ydata([])


def inOrderUpdate(root, i):
    if root is not None:
        update(root, i)
        inOrderUpdate(root.next, i)

def intersect(p, i):
    # check whether a new parabola at point p intersect with arc i
    if (i is None): return False, None
    if (i.vertex.x == p.x): return False, None

    a = 0.0
    b = 0.0

    if i.prev is not None:
        a = (intersection(i.prev.vertex, i.vertex, 1.0*p.x)).y
    if i.next is not None:
        b = (intersection(i.vertex, i.next.vertex, 1.0*p.x)).y

    if (((i.prev is None) or (a <= p.y)) and ((i.next is None) or (p.y <= b))):
        py = p.y
        px = 1.0 * ((i.vertex.x)**2 + (i.vertex.y-py)**2 - p.x**2) / (2*i.vertex.x - 2*p.x)
        res = Point(px, py)
        return True, res
    return False, None



def frontInsert(p):
    global root
    #If its the first point (most leftone)
    if root is None:
        root = Parabola(p)
    else:
        #Find current parabola at height p.y
        i = root
        while i is not None:
            flag, z = intersect(p, i)
            if flag:
                #Create extra parabola if necesary
                flag, zz = intersect(p, i.next)
                if (i.next is not None) and (not flag):
                    i.next.prev = Parabola(i.vertex, prev=i, next=i.next)
                    i.next = i.next.prev
                else:
                    i.next = Parabola(i.vertex, prev=i)
                # i.next.lowerLimit = i.lowerLimit

                i.next.prev = Parabola(p, prev=i, next=i.next)
                i.next = i.next.prev

                i = i.next

                # i.prev.upperLimit = i.lowerLimit = z
                # i.next.lowerLimit = i.upperLimit = z

                return
            i = i.next




def processPoint():
    p = Q.extractMin()
    lastAdded = p

    frontInsert(p)

def animationFrame(i):
    traversal.set_xdata(i)


    if (len(Q.heap) > 0):
        if (Q.getMin().x < i):
            processPoint()

    inOrderUpdate(root, i)


    return traversal


animation = FuncAnimation(fig, func=animationFrame, frames=np.arange(0, size*1.5, step), interval=1)


plt.show()