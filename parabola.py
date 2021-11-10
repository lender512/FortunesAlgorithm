import numpy as np
import matplotlib.pyplot as plt
from point import *
from segment import *

class Parabola():
    prev = None
    next = None
    e = None
    lowerSegment = None
    upperSegment = None

    initU = ()
    initL = ()
    parX = []
    parY = []
    foundU = False
    endU = False
    foundL = False
    endL = False
    leaf = True
    vertexU = None
    vertexL = None
    visited = False
    haveSegment = True
    finishedUpper = False
    finishedLower = False

    def __init__(self, vertex, lowerLimit = Point(0,0), upperLimit = Point(0,1), prev = None, next = None):
        self.vertex = vertex
        # self.color=np.random.rand(3,)
        self.color = (0, 0, 0)
        self.parUpper, = plt.plot(self.parX, self.parY, c=self.color)
        self.parLower, = plt.plot(self.parX, self.parY, c=self.color)
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.lowerStart = self.upperStart = Point(1,1)
        self.prev = prev
        self.next = next
        self.lowerSegment = None
        self.upperSegment = None

    def __lt__(self, other):
        return self.lowerLimit[1] < other.upperLimit[1]
    def __gt__(self, other):
        return self.upperLimit[1] > other.lowerLimit[1]
    def __ne__(self, other):
        return self.vertex != other.vertex