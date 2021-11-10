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

    def __gt__(self, other):
        return self.x > other.x
    def __ge__(self, other):
        return self.x >= other.x
    def __lt__(self, other):
        return self.x < other.x
    def __le__(self, other):
        return self.x <= other.x
    def __ne__(self, other):
        return self.x != other.x or self.y != other.y