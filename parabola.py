import numpy as np
import matplotlib.pyplot as plt

class Parabola():
    parX = []
    parY = []
    found = False
    leaf = True

    def __init__(self, vertex, lowerLimit = (0,0), upperLimit = (0,1), mostRight = 1):
        self.vertex = vertex
        self.color=np.random.rand(3,)
        self.parUpper, = plt.plot(self.parX, self.parY, c=self.color)
        self.parLower, = plt.plot(self.parX, self.parY, c=self.color)
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.mostRight = mostRight

    def __lt__(self, other):
        return self.lowerLimit[1] < other.upperLimit[1]
    def __gt__(self, other):
        return self.upperLimit[1] > other.lowerLimit[1]
    def __ne__(self, other):
        return self.vertex != other.vertex