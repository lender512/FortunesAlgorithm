import matplotlib.pyplot as plt

class Event:
    x = 0.0
    point = None
    parabola = None
    valid = True
    
    def __init__(self, x, point, parabola):
        self.x = x
        self.point = point
        self.parabola = parabola
        self.valid = True