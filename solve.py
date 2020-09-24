#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy
import re
import time
import string
import pyflowsolver

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
                return None
        return list(dots.values())

class FakeArg:
    def __init__(self):
        self.quiet = False
        self.display_cycles = False
        self.display_color = False

class FlowSolver:
    def __init__(self, size):
        self.paths = []
        self.heads = []
        self.size = size
        self.grid = defaultdict(lambda: 0)
        self.solved_names = []
        pass

    def add_pair(self, pair, name):
        for p in pair:
            self.heads.append((p, name))
            self.grid[p] = name

    def solve(self):
        # convert to pyflowsolver's format
        square = [ ['.'] * self.size for _ in range(self.size) ]
        for ((x,y), name) in self.heads:
            square[x][y] = name[0].upper()
        puzz_input = '\n'.join(''.join(L) for L in square)
        #print(puzz_input)

        options = FakeArg()
        puzzle, colors = pyflowsolver.parse_puzzle(options, puzz_input)

        color_var, dir_vars, num_vars, clauses, reduce_time = \
            pyflowsolver.reduce_to_sat(options, puzzle, colors)

        sol, decoded, repairs, solve_time = pyflowsolver.solve_sat(
                options, puzzle, colors, color_var, dir_vars, clauses)

        if not isinstance(sol, list):
            return False

        zeros = self.make2dzeros(self.size)
        for (point, name) in self.heads:
            run, is_cycle = pyflowsolver.make_path(decoded, zeros, point[0], point[1])
            if len(run) >= 1:
                #print(run)
                self.paths.append(run[:])

    def is_solved(self):
        return bool(self.paths)

    def make2dzeros(self, sz):
        return [ [0] * sz for _ in range(sz) ]


class GridInputter:
    # fractions of the screen that the grid takes up
    START_X = 0.0
    START_Y = 0.22
    END_X = 1.0
    END_Y = 0.78

    GEV_MULTIPLIER = 17.4

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

        #print(f"{x,y} becomes {px,py}")
        return round(px), round(py)


if __name__ == '__main__':
    gr = GridReader('screen.png')
    grid_size = gr.grid_size

    pairs = gr.read_dots()

    fs = FlowSolver(grid_size)
    for pair, name in zip(pairs, string.ascii_lowercase):
        fs.add_pair(pair, name)
    print()
    try:
        fs.solve()
    except:
        print(fs)
        print('heads', fs.heads)
        print('paths', fs.paths)
        raise

    if fs.is_solved():
        gi = GridInputter(grid_size)
        for path in fs.paths:
            path = gi.simplify_path(path)
            print(fs.grid[path[0]])
            for i in range(len(path)-1):
                print(f"{path[i]} -> {path[i+1]}")
                gi.input_swipe( path[i], path[i+1] , time=200)
