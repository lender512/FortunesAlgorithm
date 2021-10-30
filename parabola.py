import numpy as np
import matplotlib.pyplot as plt

class Parabola():
    parX = []
    parY = []
    found = False

    def __init__(self, lowerLimit = 0, upperLimit = 0):
        color=np.random.rand(3,)
        self.parUpper, = plt.plot(self.parX, self.parY, c=color)
        self.parLower, = plt.plot(self.parX, self.parY, c=color)
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit