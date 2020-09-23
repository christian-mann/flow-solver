#!/usr/bin/env python3
from collections import defaultdict
import pdb

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
        self.solved_names = []
        pass

    def add_pair(self, pair, name):
        self.pairs.append((pair, name))
        for p in pair:
            self.grid[p] = name

    def point_add(self, p1, p2):
        return (p1[0] + p2[0], p1[1] + p2[1])

    def on_boundary(self, point):
        # for now we just say whether it's truly on the boundary of the grid
        # later we'll add in the boundary including already painted sections
        x,y = point
        return x in (0, self.size-1) or y in (0, self.size-1)

    def on_effective_boundary(self, point):
        # either on the boundary itself
        # OR all points leading to boundary are painted
        for direction in (1,0), (-1,0), (0,1), (0,-1):
            p = point
            while True:
                p = self.point_add(p, direction)
                if not self.in_grid(p):
                    # we've escaped, effective boundary in this direction
                    return True
                if not self.grid[p]:
                    # not painted, not boundary in this direction
                    break

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
            return [start]
        # look for boundary path connecting these points
        for point in self.free_neighbors(start):
            if point in prepath:
                continue
            if not self.on_effective_boundary(point):
                continue
            subpath = self.get_path(point, end, prepath=prepath+[start])
            if subpath is not None:
                return [start] + subpath

    def paint(self, path, name):
        for p in path:
            self.grid[p] = name

    def remove_loops(self, path):
        # look for spots where the path goes in a square where it could go straight
        skip = []
        for i in range(len(path) - 3):
            p1, p2, p3, p4 = path[i:i+4]
            if p4 in self.neighbors(p1):
                skip.append(p2)
                skip.append(p3)
        return [p for p in path if p not in skip]


    def solve(self):
        # look for pairs of points that are on the boundary
        found_path = True
        while found_path:
            found_path = False
            for (p1, p2), name in self.pairs:
                assert self.grid[p1] == self.grid[p2] == name

                if name in self.solved_names:
                    continue
                if self.on_effective_boundary(p1) and self.on_effective_boundary(p2):
                    print( f"{name} = {p1,p2} on boundary" )
                    print( f"Finding path for {name}" )

                    # temporarily unmark endpoints so the path algorithm works
                    self.grid[p1] = self.grid[p2] = 0
                    path = self.get_path(p1, p2)
                    if path:
                        path = self.remove_loops(path)
                        print( f"Path is: {path}" )
                        found_path = True
                        self.paint(path, name)
                        self.solved_names.append(name)
                    self.grid[p1] = self.grid[p2] = name


    def __str__(self):
        out = ''
        for y in range(self.size):
            line = ''
            for x in range(self.size):
                if self.grid[x,y]:
                    line += self.grid[x,y][0].upper()
                else:
                    line += ' '
            out += line + '\n'
        return out

if __name__ == '__main__':
    fs = FlowSolver(GRID_SIZE)
    for (pair, name) in INPUT:
        fs.add_pair(pair, name)
    fs.solve()
    print(fs)
