
class Point:
    def __init__(self, point):
        self.point = point
        self.x = point[0]
        self.y = point[1]


class MatchingPoint(Point):
    def __init__(self, point, value):
        super().__init__(point)
        self.id = None
        self.value = value
        self.cluster = None

    def setCluster(self, cluster):
        self.cluster = cluster
