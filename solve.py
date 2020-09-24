#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy
import re
import time
import string

import cv2
import numpy as np

from adb_shell.adb_device import AdbDeviceTcp

import pdb
from collections import Counter
import sys

# could do SAT solver... but let's try to make our own algorithm

class GridReader:
    START_X = 0.0
    START_Y = 0.22
    END_X = 1.0
    END_Y = 0.78

    def __init__(self, fname):
        self.img = cv2.imread(fname)
        self.sheight, self.swidth = self.img.shape[:2]
        self.grid_size = self.determine_grid_size()

    def is_same_color(self, k, k2):
        if abs(k[0] - k2[0]) > 2: return False
        if abs(k[1] - k2[1]) > 2: return False
        if abs(k[2] - k2[2]) > 2: return False
        return True

    def determine_grid_size(self):
        gridline_color = self.img[
            round(self.sheight * self.START_Y + 5),
            100]

        num_gridlines = 0
        in_gridline = False
        # try and count the vertical lines
        py = round(self.sheight * self.START_Y + 10)
        for px in range(self.swidth):
            k = self.img[py,px]
            if self.is_same_color(k, gridline_color):
                if in_gridline:
                    # already in gridline, we don't care
                    pass
                else:
                    # new gridline
                    num_gridlines += 1
                    in_gridline = True
            else:
                if in_gridline:
                    # gridline ended
                    in_gridline = False
                else:
                    # already not in a gridline
                    pass
        return num_gridlines - 1

    def read_dots(self):
        # produces List[Pair[Coordinate]]

        # get the color at the center of each gridline
        dx = self.swidth / self.grid_size
        dy = self.sheight * (self.END_Y - self.START_Y) / self.grid_size

        dots = defaultdict(list)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                px = x * dx + dx/2
                py = y * dy + dy/2 + self.START_Y * self.sheight

                px = round(px)
                py = round(py)

                k = tuple(self.img[py,px])
                k = tuple(val if val >= 30 else 0 for val in k)

                if k != (0,0,0):
                    # dot
                    dots[k].append((x,y))

        # examine the dots array and make sure it makes sense
        for color, coords in dots.items():
            if len(coords) != 2:
                print(f"Error: more than 2 dots of color {color}: {coords}")
        return list(dots.values())



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

    def step_two_adjacent_heads(self):
        for head in self.heads:
            point, name = head
            neighbors = self.neighbors(point)

            for nei in neighbors:
                if self.is_head(nei) and self.grid[nei] == name:
                    # dots connected - remove heads
                    self.heads.remove( (point, name) )
                    self.heads.remove( (nei, name) )
                    # canonicalize paths
                    self.paths[point] += self.paths[nei][::-1]
                    del self.paths[nei]
                    # mark color as solved
                    self.solved_names.append(name)
                    return name
        return None

    def step_only_one_option(self):
        for head in self.heads:
            point, name = head
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
                
                return name
        return None

    def step_boundary_path(self):
        for (start, name) in self.heads:
            if self.on_effective_boundary(start, name):
                # look for other end
                end, name2 = [(end,name2) for (end,name2) in self.heads if name == name2 and start != end][0]
                paths = list(self.get_boundary_paths(start, end, name))
                if paths:
                    # pick the shortest one
                    path = min(paths, key=len)

                    # paint grid
                    for p in path:
                        self.grid[p] = name
                    # remove heads
                    self.heads.remove( (start, name) )
                    self.heads.remove( (end, name2) )
                    # combine paths
                    self.paths[start] = self.paths[start] + path + self.paths[end][::-1]
                    del self.paths[end]
                    # mark name as solved
                    self.solved_names.append(name)
                    return name
        return None

    def step_rescue_square(self):
        # looking for a square that can only be reached by one head
        for head in self.heads:
            point, name = head
            for neigh in self.free_neighbors(point):
                open_neighbors = 0
                other_head_adjacent = False
                for nei2 in self.neighbors(neigh):
                    if nei2 == point:
                        # of course we're our neighbor's neighbor
                        pass
                    elif not self.grid[nei2]:
                        open_neighbors += 1
                    elif self.is_head(nei2):
                        other_head_adjacent = True
                if open_neighbors == 1 and not other_head_adjacent:
                    # this head must go to this point, nothing else will
                    # paint grid
                    self.grid[neigh] = name
                    # update head
                    self.heads.remove(head)
                    self.heads.append( (neigh, name) )
                    # update path
                    self.paths[neigh] = self.paths[point] + [neigh]
                    del self.paths[point]

                    return name
        return None


    def solve(self):
        # look for a head with only one open neighbor
        found = True
        while found:
            print(self)
            found = False

            # look for two heads next to each other
            if name := self.step_two_adjacent_heads():
                found = True
                print(f"two adjacent heads for {name}")
                continue

            # look for a head that can only go one direction
            if name := self.step_only_one_option():
                found = True
                print(f"only one option for {name}")
                continue

            # look for a head that can travel to its counterpart solely by boundary tiles
            if name := self.step_boundary_path():
                found = True
                print(f"found boundary path for {name}")
                continue
            
            if name := self.step_rescue_square():
                found = True
                print(f"found square that could only be reached by {name}")
                continue


    def is_solved(self):
        return not self.heads and all(self.grid.values())

    def on_boundary(self, point):
        # for now we just say whether it's truly on the boundary of the grid
        # later we'll add in the boundary including already painted sections
        x,y = point
        return x in (0, self.size-1) or y in (0, self.size-1)

    def on_effective_boundary(self, point, name):
        # either on the boundary itself
        # or one of its neighbors is solved
        if self.on_boundary(point): return True
        if any(self.grid[p] in self.solved_names for p in self.neighbors(point)): return True
        if any(self.grid[p] in self.solved_names for p in self.diagonals(point)): return True

        # OR all points leading to boundary are painted with a DIFFERENT name
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
                if self.grid[p] == name:
                    # painted the same color, this is considered unpainted, so not boundary
                    break

    def get_boundary_paths(self, start, end, name, prepath=None):
        if prepath is None:
            prepath = []

        if start == end:
            yield [start]
        # look for boundary path connecting these points
        for point in self.neighbors(start):
            if self.grid[point] and self.grid[point] != name:
                # already painted by another color
                continue
            if point in prepath:
                continue
            if not self.on_effective_boundary(point, name):
                continue
            for subpath in self.get_boundary_paths(point, end, name, prepath=prepath+[start]):
                yield [start] + subpath

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

    def diagonals(self, point):
        x,y = point
        potentials = [
            (x-1,y-1),
            (x-1,y+1),
            (x+1,y-1),
            (x+1,y+1),
        ]
        return [p for p in potentials if self.in_grid(p)]

    def free_neighbors(self, point):
        return [p for p in self.neighbors(point) if not self.grid[p]]

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

        # lastly, the game wants to help us complete paths so if the last two are neighbors we remove the last one
        if len(new_path) >= 3:
            if abs(new_path[-1][0] - new_path[-2][0]) + abs(new_path[-1][1] - new_path[-2][1]) == 1:
                new_path = new_path[:-1]
        return new_path


    def get_pixels(self, point):
        # translate grid point to pixel offset
        x,y = point

        dx = self.swidth / self.grid_size
        dy = self.sheight * (self.END_Y - self.START_Y) / self.grid_size

        px = x * dx + dx/2
        py = y * dy + dy/2 + self.START_Y * self.sheight

        return round(px), round(py)


if __name__ == '__main__':
    gr = GridReader('screen.png')
    grid_size = gr.grid_size

    pairs = gr.read_dots()

    fs = FlowSolver(grid_size)
    for pair, name in zip(pairs, string.ascii_lowercase):
        fs.add_pair(pair, name)
    print(fs)
    print()
    try:
        fs.solve()
    except:
        print(fs)
        print('heads', fs.heads)
        print('paths', fs.paths)
        raise
    print(fs.is_solved())
    print(fs)

    if True or fs.is_solved():
        gi = GridInputter(grid_size)
        for path in fs.paths.values():
            path = gi.simplify_path(path)
            print(fs.grid[path[0]])
            for i in range(len(path)-1):
                print(f"{path[i]} -> {path[i+1]}")
                gi.input_swipe( path[i], path[i+1] , time=400)
