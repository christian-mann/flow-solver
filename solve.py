#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy
import re
import time

from adb_shell.adb_device import AdbDeviceTcp

import pdb

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

# level 1 from Bonus Pack
INPUT = [
    ( ( (1,3), (4,1) ), "yellow"),
    ( ( (0,3), (4,3) ), "green"),
    ( ( (1,1), (3,1) ), "blue"),
    ( ( (3,3), (4,4) ), "red"),
]
GRID_SIZE = 5

# level 2 from Bonus Pack 5x5 tier
INPUT = [
    ( ( (0,1), (4,4) ), "green"),
    ( ( (0,2), (3,2) ), "yellow"),
    ( ( (3,1), (1,2) ), "red"),
    ( ( (2,2), (1,3) ), "blue"),
]
GRID_SIZE = 5

# level 1 from Bonus Pack 9x9 tier
INPUT = [
    ( ( (0,0), (8,4) ), "green"),
    ( ( (0,1), (1,8) ), "red"),
    ( ( (2,1), (7,6) ), "maroon"),
    ( ( (0,2), (3,3) ), "yellow"),
    ( ( (5,2), (5,5) ), "blue"),
    ( ( (1,5), (8,6) ), "orange"),
    ( ( (2,5), (4,7) ), "sky blue"),
    ( ( (3,5), (7,7) ), "purple"),
    ( ( (7,1), (8,5) ), "vink"),
]
GRID_SIZE = 9

# level 2 from Bonus Pack 9x9 tier
INPUT = [
    ( ( (1,0), (1,6) ), "vink"),
    ( ( (2,0), (5,5) ), "yellow"),
    ( ( (3,0), (3,3) ), "green"),
    ( ( (3,1), (2,3) ), "orange"),
    ( ( (5,0), (4,4) ), "blue"),
    ( ( (6,0), (0,7) ), "red"),
    ( ( (1,5), (6,1) ), "sky"),
    ( ( (7,1), (1,7) ), "maroon"),
]


class FlowSolver:
    def __init__(self, size):
        self.paths = {}
        self.heads = []
        self.size = size
        self.grid = defaultdict(lambda: 0)
        self.solved_names = []
        pass

    def add_pair(self, pair, name):
        for p in pair:
            self.paths[p] = [p]
            self.heads.append((p, name))
            self.grid[p] = name

    # adds two points together -- this is used to get a point's neighbor
    def point_add(self, p1, p2):
        return (p1[0] + p2[0], p1[1] + p2[1])

    def is_head(self, point):
        return any(p == point for (p,n) in self.heads)

    def solve(self):
        # look for a head with only one open neighbor
        found = True
        while found:
            found = False
            for head in self.heads:
                point, name = head
                neighbors = self.neighbors(point)

                # first look for neighboring head
                for nei in neighbors:
                    if self.is_head(nei) and self.grid[nei] == name:
                        # dots connected - remove heads
                        self.heads.remove( (point, name) )
                        self.heads.remove( (nei, name) )
                        # canonicalize paths
                        self.paths[point] += self.paths[nei][::-1]
                        del self.paths[nei]
                        found = True

                if not found:
                    # look for somewhere to expand
                    free_nei = self.free_neighbors(point)
                    if len(free_nei) == 1:
                        # paint grid
                        self.grid[free_nei[0]] = name

                        # update head
                        self.heads.remove(head)
                        head = (free_nei[0], name)
                        self.heads.append(head)

                        # update path
                        self.paths[free_nei[0]] = self.paths[point] + [free_nei[0]]
                        del self.paths[point]

                        found = True

    def is_solved(self):
        return not self.heads and all(self.grid.values())

    def on_boundary(self, point):
        # for now we just say whether it's truly on the boundary of the grid
        # later we'll add in the boundary including already painted sections
        x,y = point
        return x in (0, self.size-1) or y in (0, self.size-1)

    def on_effective_boundary(self, point):
        # either on the boundary itself
        # or one of its neighbors is solved
        if self.on_boundary(point): return True
        if any(self.grid[p] in self.solved_names for p in self.neighbors(point)): return True
        if any(self.grid[p] in self.solved_names for p in self.diagonals(point)): return True

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

    def remove_loops(self, path):
        # look for spots where the path goes in a square where it could go straight
        skip = []
        for i in range(len(path) - 3):
            p1, p2, p3, p4 = path[i:i+4]
            if p4 in self.neighbors(p1):
                skip.append(p2)
                skip.append(p3)
        return [p for p in path if p not in skip]


    def __str__(self):
        out = ''
        for y in range(self.size):
            line = ''
            for x in range(self.size):
                if self.grid[x,y]:
                    if self.is_head((x,y)):
                        line += self.grid[x,y][0].upper()
                    else:
                        line += self.grid[x,y][0].lower()
                else:
                    line += ' '
            out += line + '\n'
        return out


class GridInputter:
    # fractions of the screen that the grid takes up
    START_X = 0.0
    START_Y = 0.22
    END_X = 1.0
    END_Y = 0.78

    def __init__(self, grid_size, host='localhost', port=5555):
        self.dev = AdbDeviceTcp(host, port, default_transport_timeout_s=9.)
        self.dev.connect()
        self.swidth, self.sheight = self.screen_dimensions()
        self.grid_size = grid_size

    def __del__(self):
        self.dev.close()

    def screen_dimensions(self):
        line = self.dev.shell('wm size')
        result = re.match(r"Physical size: (?P<height>\d+)x(?P<width>\d+)\n", line)
        return int(result.group('width')), int(result.group('height'))

    def input_swipe(self, p1, p2, time=100):
        start_pix = self.get_pixels(p1)
        end_pix = self.get_pixels(p2)

        self.dev.shell(f'input swipe {start_pix[0]} {start_pix[1]} {end_pix[0]} {end_pix[1]} {time}')

    def simplify_path(self, path):
        path = path[:]
        new_path = []
        i = 0
        while i + 2 < len(path):
            if path[i+0][0] == path[i+1][0] == path[i+2][0]:
                # we can remove the middle one
                path.remove(path[i+1])
            elif path[i+0][1] == path[i+1][1] == path[i+2][1]:
                path.remove(path[i+1])
            else:
                new_path.append(path[i+0])
                i += 1
        new_path += path[i:]
        return new_path


    def get_pixels(self, point):
        # translate grid point to pixel offset
        x,y = point

        dx = self.swidth / self.grid_size
        dy = self.sheight * (self.END_Y - self.START_Y) / self.grid_size

        px = x * dx + dx/2
        py = y * dy + dy/2 + self.START_Y * self.sheight

        print(f'{x,y} becomes {px,py}')
        return round(px), round(py)


if __name__ == '__main__':
    fs = FlowSolver(GRID_SIZE)
    for (pair, name) in INPUT:
        fs.add_pair(pair, name)
    print(fs)
    print()
    fs.solve()
    print(fs.is_solved())
    print(fs)

    gi = GridInputter(GRID_SIZE)
    for path in fs.paths.values():
        path = gi.simplify_path(path)
        print(fs.grid[path[0]])
        for i in range(len(path)-1):
            #print(f"{path[i]} -> {path[i+1]}")
            gi.input_swipe( path[i], path[i+1] , time=50)
        time.sleep(0.5)
