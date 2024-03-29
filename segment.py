import matplotlib.pyplot as plt

class Segment:
    start = None
    end = None
    done = False
    
    def __init__(self, p):
        self.start = p
        self.end = None
        self.done = False
        self.points, = plt.plot([], [], c=(0,0,0), linewidth=.7)

    def update(self, a, b):
        self.points.set_data([a.x, b.x], [a.y, b.y])
        self.start = a
        self.end = b

    def finish(self, p):
        if self.done: return
        self.points.set_data([self.start.x, p.x], [self.start.y, p.y])
        self.end = p
        self.done = True    