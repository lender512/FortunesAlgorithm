import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from minHeap import *
from parabola import *
from point import *
from event import *
import math

fig, ax = plt.subplots()

step = 0.00001
size = 1
n = 100

ax.set_xlim(0, size), ax.set_ylim(0, size)

x = np.sort(np.random.choice(np.arange(0, size, step*100), size=n))
y = np.random.choice(np.arange(0, size, step*100), size=n)

intList = []

for i in range(n):
    intList.append((int(x[i]*1000), int(y[i]*1000)))

print(intList)

#Priorityqueue of points in the plane
Q = MinHeap()
E = MinHeap()

#Insert in priorityQueue
for i in range(n):
    p = Point(x[i], y[i])
    Q.insertKey(p)

root = None

plt.scatter(x, y, s=10)

#Sweeping line
traversal, = plt.plot((0,0), (0,1))

def getIntersection(p0, p1, l, lower):
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

        a = 1.0/z0 - 1.0/z1
        b = -2.0 * (p0.y/z0 - p1.y/z1)
        c = 1.0 * (p0.y**2 + p0.x**2 - l**2) / z0 - 1.0 * (p1.y**2 + p1.x**2 - l**2) / z1

        if lower:
            py = 1.0 * (-b-math.sqrt(b*b - 4*a*c)) / (2*a)
        else:
            py = 1.0 * (-b+math.sqrt(b*b - 4*a*c)) / (2*a)

    px = 1.0 * (p.x**2 + (p.y-py)**2 - l**2) / (2*p.x-2*l)

    #Avoid intersection to be far away and couse compelxity issues
    if px < -0.75:
        px = -0.75
    return Point(px, py) 

def update(parabola, i):
    xCoord = parabola.vertex.x
    yCoord = parabola.vertex.y
    parabola.lowerStart = Point(i, yCoord)
    parabola.upperStart = Point(i, yCoord)
    p = i-xCoord

    if parabola.e is None or parabola.e.valid:

        if parabola.next is not None:
            if parabola.next.vertex.x > parabola.vertex.x:
                # The next parabola is over the current parabola
                intersection = getIntersection(parabola.vertex, parabola.next.vertex, i, True)

                if intersection.y < parabola.vertex.y:
                    #The upper part of the parabola collides with the upper part of the next parabola
                    parabola.lowerStart = intersection
                    
                    parabolaX = np.arange(parabola.lowerLimit.x, parabola.lowerStart.x, step)
                    parabolaY = -np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
                    
                    parabola.parLower.set_data(parabolaX, parabolaY)
                    parabola.parUpper.set_data((),())
                else:
                    #The upper part of the parabola collides with the lower part of the next parabola
                    parabola.upperLimit = intersection
                    parabolaX = np.arange(parabola.lowerLimit.x, parabola.lowerStart.x, step)
                    parabolaY = -np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
                    parabola.parLower.set_data(parabolaX, parabolaY)
                    parabolaX = np.arange(parabola.upperLimit.x, parabola.upperStart.x, step)
                    parabolaY = np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
                    parabola.parUpper.set_data(parabolaX, parabolaY)


            elif parabola.next.vertex.x < parabola.vertex.x:
                # The next parabola is under the current parabola
                parabola.upperLimit = getIntersection(parabola.vertex, parabola.next.vertex, i, True)
                parabolaX = np.arange(parabola.upperLimit.x, parabola.upperStart.x, step)
                parabolaY = np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord

                parabola.parUpper.set_data(parabolaX, parabolaY)
                parabola.parLower.set_data((),())
                
            if parabola.upperSegment is not None and not parabola.upperSegment.done:
                parabola.upperSegment.update(parabola.upperSegment.start, getIntersection(parabola.vertex, parabola.next.vertex, i, True))
        
        if parabola.prev is not None:
            if parabola.prev.vertex.x > parabola.vertex.x:
                # The prev parabola is over the current parabola
                intersection = getIntersection(parabola.vertex, parabola.prev.vertex, i, False)
                
                if intersection.y > parabola.vertex.y:
                    #The lower part of the parabola collides with the lower part of the next parabola
                    parabola.upperStart = intersection
                    parabolaX = np.arange(parabola.upperLimit.x, parabola.upperStart.x, step)
                    parabolaY = np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord

                    parabola.parUpper.set_data(parabolaX, parabolaY)
                    parabola.parLower.set_data((),())
                else:
                    #The lower part of the parabola collides with the upper part of the prev parabola
                    parabola.lowerLimit = intersection
                    parabolaX = np.arange(parabola.upperLimit.x, parabola.upperStart.x, step)
                    parabolaY = np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
                    if (abs(parabola.lowerStart.x-parabola.upperStart.x) == 0):
                        parabola.parUpper.set_data(parabolaX, parabolaY)
                    else:
                        parabola.parUpper.set_data((), ())
                    parabolaX = np.arange(parabola.lowerLimit.x, parabola.lowerStart.x, step)
                    parabolaY = -np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
                    parabola.parLower.set_data(parabolaX, parabolaY)


            elif parabola.prev.vertex.x < parabola.vertex.x:
                # The prev parabola is under the current parabola
                parabola.lowerLimit = getIntersection(parabola.vertex, parabola.prev.vertex, i, False)
                parabolaX = np.arange(parabola.lowerLimit.x, parabola.lowerStart.x, step)
                parabolaY = -np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord

                parabola.parLower.set_data(parabolaX, parabolaY)

            if parabola.lowerSegment is not None and not parabola.lowerSegment.done:
                parabola.lowerSegment.update(parabola.lowerSegment.start, getIntersection(parabola.vertex, parabola.prev.vertex, i, False))

        if parabola.prev is None and parabola.next is None:

            parabolaX = np.arange(parabola.upperLimit.x, parabola.upperStart.x, step)
            parabolaY = np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
            parabola.parUpper.set_data(parabolaX, parabolaY)

            parabolaX = np.arange(parabola.lowerLimit.x, parabola.lowerStart.x, step)
            parabolaY = -np.sqrt(-(2*p*(parabolaX-i+p/2)))+yCoord
            parabola.parLower.set_data(parabolaX, parabolaY)


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
        a = (getIntersection(i.prev.vertex, i.vertex, 1.0*p.x, True)).y
    if i.next is not None:
        b = (getIntersection(i.vertex, i.next.vertex, 1.0*p.x, True)).y

    if (i.prev is None or a <= p.y) and (i.next is None or p.y <= b):
        py = p.y
        px = 1.0 * ((i.vertex.x)**2 + (i.vertex.y-py)**2 - p.x**2) / (2*i.vertex.x - 2*p.x)
        res = Point(px, py)
        return True, res
    return False, None

#Return the right most point of a given circle
def circle(a, b, c):
    # check if bc is a "right turn" from ab
    if ((b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y)) > 0: return False, None, None

    # Joseph O'Rourke, Computational Geometry in C (2nd ed.) p.189
    A = b.x - a.x
    B = b.y - a.y
    C = c.x - a.x
    D = c.y - a.y
    E = A*(a.x + b.x) + B*(a.y + b.y)
    F = C*(a.x + c.x) + D*(a.y + c.y)
    G = 2*(A*(c.y - b.y) - B*(c.x - b.x))

    if (G == 0): return False, None, None # Points are co-linear

    # point o is the center of the circle
    ox = 1.0 * (D*E - B*F) / G
    oy = 1.0 * (A*F - C*E) / G

    # o.x plus radius equals max x coord
    x = ox + math.sqrt((a.x-ox)**2 + (a.y-oy)**2)
    o = Point(ox, oy)
        
    return True, x, o

#Function to search for circular events
def searchCircleEvent(parabola, x0):
        # look for a new circle event for arc parabola
        if (parabola.e is not None) and (parabola.e.x != x0):
            parabola.e.valid = False
        parabola.e = None

        if (parabola.prev is None) or (parabola.next is None): return

        flag, x, o = circle(parabola.prev.vertex, parabola.vertex, parabola.next.vertex)
        if flag and (x > x0):
            parabola.e = Event(x, o, parabola)
            E.insertKey(parabola.e)

def frontInsert(p, l):
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
                    i.next.prev.haveSegment = False
                    i.next = i.next.prev
                else:
                    i.next = Parabola(i.vertex, prev=i)
                i.next.upperSegment = i.upperSegment

                i.next.prev = Parabola(p, prev=i, next=i.next)
                i.next = i.next.prev

                i = i.next
                
                seg = Segment(z)
                # output.append(seg)
                i.prev.upperSegment = i.lowerSegment = seg

                seg = Segment(z)
                # output.append(seg)
                i.next.lowerSegment = i.upperSegment = seg

                #Check for circle events
                searchCircleEvent(i, p.x)
                searchCircleEvent(i.prev, p.x)
                searchCircleEvent(i.next, p.x)
                return
            i = i.next

def processPoint(l):
    p = Q.extractMin()

    frontInsert(p, l)

def processCircle():
    event = E.extractMin()

    if event.valid:

        s = Segment(event.point)

        #Remove linked parabola
        parabola = event.parabola
        if parabola.prev is not None:
            parabola.prev.next = parabola.next
            parabola.prev.upperSegment = s
        if parabola.next is not None:
            parabola.next.prev = parabola.prev
            parabola.next.lowerSegment = s

        if parabola.lowerSegment is not None: 
            parabola.lowerSegment.finish(event.point)
            parabola.parLower.set_data((),())
        if parabola.upperSegment is not None: 
            parabola.upperSegment.finish(event.point)
            parabola.parUpper.set_data((),())

        if parabola.prev is not None: searchCircleEvent(parabola.prev, event.x)
        if parabola.next is not None: searchCircleEvent(parabola.next, event.x)

def animationFrame(i):
    traversal.set_xdata(i)

    if (len(E.heap) > 0) and E.getMin().x < i:
        processCircle()
    if (len(Q.heap) > 0) and Q.getMin().x < i:
        processPoint(i)


    inOrderUpdate(root, i)

    return traversal


animation = FuncAnimation(fig, func=animationFrame, frames=np.arange(0, size*1.5, step*100), interval=5)


plt.show()