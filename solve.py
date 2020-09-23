#!/usr/bin/env python3
from collections import defaultdict

# given: List[Pair[Coordinate]], Dimensions
# output: List[Path] = List[List[Coordinate]]


# could do SAT solver... but let's try to make our own algorithm

# (x,y) where 0,0 is top-left
# level 1 from Classic Pack
INPUT = [
    ( ( (0,0), (1,4) ), "red" ),
    ( ( (2,0), (1,3) ), "green" ),
    ( ( (2,1), (2,4) ), "blue" ),
    ( ( (4,0), (3,3) ), "yellow" ),
    ( ( (4,1), (3,4) ), "orange" ),
]
GRID_SIZE = 5


class FlowSolver:
    def __init__(self, size):
        self.pairs = []
        self.size = size
        self.grid = defaultdict(lambda: 0)
        pass

    def add_pair(self, pair, name):
        self.pairs.append((pair, name))
        for p in pair:
            self.grid[p] = name

    def on_boundary(self, point):
        # for now we just say whether it's truly on the boundary of the grid
        # later we'll add in the boundary including already painted sections
        x,y = point
        return x in (0, self.size-1) or y in (0, self.size-1)

    def in_grid(self, point):
        x,y = point
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbors(self, point):
        x,y = point
        potentials = [
            (x-1,y),
            (x+1,y),
            (x,y-1),
            (x,y+1)
        ]
        return [p for p in potentials if self.in_grid(p)]

    def free_neighbors(self, point):
        return [p for p in self.neighbors(point) if not self.grid[p]]

    def get_path(self, start, end, prepath=None):
        if prepath is None:
            prepath = []

        if start == end:
            return []
        # look for boundary path connecting these points
        for point in self.free_neighbors(start):
            if point in prepath:
                continue
            if not self.on_boundary(point):
                continue
            subpath = self.get_path(point, end, prepath=prepath+[start])
            if subpath is not None:
                return [start] + subpath

    def paint(self, path, name):
        for p in path:
            self.grid[p] = name

    def solve(self):
        # look for pairs of points that are on the boundary
        for (p1, p2), name in self.pairs:
            if self.on_boundary(p1) and self.on_boundary(p2):
                print( f"{name} = {p1,p2} on boundary" )
                print( "Finding path for {name}" )

                # temporarily unmark p2 so the path algorithm works
                prev_p2 = self.grid[p2]
                self.grid[p2] = 0
                path = self.get_path(p1, p2)
                self.grid[p2] = prev_p2
                self.paint(path, name)

                print( f"Path is: {path}" )

    def tostring(self):
        for y in range(self.size):
            line = ''
            for x in range(self.size):
                if self.grid[x,y]:
                    line += self.grid[x,y][0].upper()
                else:
                    line += ' '
            print(line)

if __name__ == '__main__':
    fs = FlowSolver(GRID_SIZE)
    for (pair, name) in INPUT:
        fs.add_pair(pair, name)
    fs.solve()
    fs.tostring()
