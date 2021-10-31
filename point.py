class Point:
    
    def __init__(self, x, y, siteEvent = True):
        self.x = x
        self.y = y
        self.siteEvent = siteEvent


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