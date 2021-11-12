import matplotlib.pyplot as plt
from point import *
from segment import *

class Parabola():
    prev = None
    next = None
    e = None
    lowerSegment = None
    upperSegment = None

    parX = []
    parY = []
    haveSegment = True
    color = ()

    def __init__(self, vertex, lowerLimit = Point(0,0), upperLimit = Point(0,1), prev = None, next = None):
        self.vertex = vertex
        # self.color=np.random.rand(3,)
        self.color = (0.5, 0, 0)
        self.parUpper, = plt.plot(self.parX, self.parY, c=self.color)
        self.parLower, = plt.plot(self.parX, self.parY, c=self.color)
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.lowerStart = self.upperStart = self.vertex
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