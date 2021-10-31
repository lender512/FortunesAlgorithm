import numpy as np
import matplotlib.pyplot as plt
from point import *

class Segment:
    start = None
    end = None
    done = False
    
    def __init__(self, p):
        self.start = p
        self.end = None
        self.done = False

    def finish(self, p):
        if self.done: return
        self.end = p
        self.done = True 

class Parabola():
    prev = None
    next = None


    parX = []
    parY = []
    found = False
    leaf = True

    def __init__(self, vertex, lowerLimit = Point(0,0), upperLimit = Point(0,1), prev = None, next = None):
        self.vertex = vertex
        self.color=np.random.rand(3,)
        self.parUpper, = plt.plot(self.parX, self.parY, c=self.color)
        self.parLower, = plt.plot(self.parX, self.parY, c=self.color)
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.prev = prev
        self.next = next

    def __lt__(self, other):
        return self.lowerLimit[1] < other.upperLimit[1]
    def __gt__(self, other):
        return self.upperLimit[1] > other.lowerLimit[1]
    def __ne__(self, other):
        return self.vertex != other.vertex