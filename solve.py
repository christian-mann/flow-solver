#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy
import re
import time
import string
import pyflowsolver
import pprint

import cv2
import numpy as np

from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

import pdb
from collections import Counter
import sys

# could do SAT solver... but let's try to make our own algorithm


class GridReader:
    START_X = 0.0
    END_X = 1.0

    def __init__(self, fname):
        self.img = cv2.imread(fname)
        self.sheight, self.swidth = self.img.shape[:2]
        print(f'{self.sheight=}, {self.swidth=}')
        self.gridline_color = self.determine_gridline_color()
        print(f'{self.gridline_color=}')
        self.gwidth, self.gheight = self.determine_grid_size()
        print(f'{self.gheight=}, {self.gwidth=}')
        self.starty, self.endy = self.determine_ybound()
        print(f'{self.starty=}, {self.endy=}')
        self.startx, self.endx = self.determine_xbound()
        print(f'{self.startx=}, {self.endx=}')

    def is_same_color(self, k, k2):
        if abs(k[0] - k2[0]) > 2: return False
        if abs(k[1] - k2[1]) > 2: return False
        if abs(k[2] - k2[2]) > 2: return False
        return True

    def get_num_strings(self, pixels, target):
        # aaBBccBBBBddBBaB -> 4, since 4 groups of B
        in_gridline = False
        num_gridlines = 0
        for k in pixels:
            if self.is_same_color(k, target):
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
        return num_gridlines
                
    def determine_gridline_color(self):
        colors = [
            self.img[
                self.sheight // 2 + 3,
                self.swidth // 2 - 10 + i
            ] for i in range(200)
        ]
        #pprint.pprint(colors)
        # look for 2-3 in a row
        prev_c = None
        count = 0
        for c in colors:
            if np.array_equal(c, prev_c):
                count += 1
            else:
                if count in (3,4) and np.all(prev_c):
                    return prev_c
                else:
                    prev_c = c
                    count = 1
        raise Exception("Could not identify gridline color")

    def determine_grid_size(self):

        num_gridlines = 0
        in_gridline = False
        # try and count the vertical lines
        py = round(self.sheight * 0.5) + 3 # near the middle but not exactly
        num_vlines = self.get_num_strings(
            (self.img[py, x] for x in range(self.swidth)),
            self.gridline_color
        )
        px = round(self.swidth * 0.5) + 3
        num_hlines = self.get_num_strings(
            (self.img[y, px] for y in range(self.sheight)),
            self.gridline_color
        )
        print(f'{num_vlines=}, {num_hlines=}')
        return (num_vlines - 1, num_hlines - 1)

    def determine_ybound(self):
        # just the first and last instance of gridline_color
        first = None
        last = None
        for y in range(self.sheight):
            if self.is_same_color(self.gridline_color, self.img[y, self.swidth // 2 + 10]):
                if not first:
                    first = y
                last = y
        if not first or not last:
            raise Exception("Could not determine ybound")
        return (first, last)
    
    def determine_xbound(self):
        # just the first and last instance of gridline_color
        first = None
        last = None
        for x in range(self.swidth):
            if self.is_same_color(self.gridline_color, self.img[self.sheight // 2 + 5, x]):
                if not first:
                    first = x
                last = x
        if not first or not last:
            raise Exception("Could not determine xbound")
        return (first, last)

    def read_dots(self):
        # produces List[Pair[Coordinate]]

        # get the color at the center of each gridline
        dy = (self.endy - self.starty) / self.gheight
        dx = (self.endx - self.startx) / self.gwidth

        dots = defaultdict(list)
        for y in range(self.gheight):
            for x in range(self.gwidth):
                px = x * dx + dx/2 + self.startx
                py = y * dy + dy/2 + self.starty

                px = round(px)
                py = round(py)

                k = tuple(self.img[py,px])
                #print(f'{px=} {py=} {k=}')
                k = tuple(val if val >= 30 else 0 for val in k)

                if k != (0,0,0):
                    # dot
                    dots[k].append((y,x))

        # examine the dots array and make sure it makes sense
        for color, coords in dots.items():
            if len(coords) != 2:
                raise Exception(f"Error: more than 2 dots of color {color}: {coords}")
        return list(dots.values())

class FakeArg:
    def __init__(self):
        self.quiet = False
        self.display_cycles = False
        self.display_color = False
        self.repair_colors = False

class FlowSolver:
    def __init__(self, gwidth, gheight):
        self.paths = []
        self.heads = []
        self.gwidth = gwidth
        self.gheight = gheight
        self.grid = defaultdict(lambda: 0)
        self.solved_names = []
        pass

    def add_pair(self, pair, name):
        for p in pair:
            self.heads.append((p, name))
            self.grid[p] = name

    def solve(self):
        # convert to pyflowsolver's format
        square = [ ['.'] * self.gwidth for _ in range(self.gheight) ]
        for ((y,x), name) in self.heads:
            square[y][x] = name[0].upper()
        puzz_input = '\n'.join(''.join(L) for L in square)
        print(puzz_input)

        options = FakeArg()
        puzzle, colors = pyflowsolver.parse_puzzle(options, puzz_input)

        color_var, dir_vars, num_vars, clauses, reduce_time = \
            pyflowsolver.reduce_to_sat(options, puzzle, colors)

        sol, decoded, repairs, solve_time = pyflowsolver.solve_sat(
                options, puzzle, colors, color_var, dir_vars, clauses)

        #print(f'{decoded=}, {repairs=}, {solve_time=}')
        #pprint.pprint(decoded)
        if not isinstance(sol, list):
            return False

        zeros = self.make2dzeros(self.gheight, self.gwidth)
        #print(f'{self.heads=}')
        for ((y,x), name) in self.heads:
            run, is_cycle = pyflowsolver.make_path(decoded, zeros, y, x)
            if len(run) >= 1:
                #print(run)
                self.paths.append(run[:])

    def is_solved(self):
        return bool(self.paths)

    def make2dzeros(self, height, width):
        return [ [0] * width for _ in range(height) ]


class GridInputter:
    # fractions of the screen that the grid takes up
    START_X = GridReader.START_X
    END_X = GridReader.END_X

    def __init__(self, gr, *, host='192.168.1.179', port=5555):
        with open('adbkey') as f:
            priv = f.read()
        with open('adbkey.pub') as f:
            pub = f.read()
        signer = PythonRSASigner(pub, priv)
        self.dev = AdbDeviceTcp(host, port, default_transport_timeout_s=9.)
        self.dev.connect(rsa_keys=[signer])
        self.gr = gr
        self.swidth, self.sheight = gr.swidth, gr.sheight
        self.gwidth, self.gheight = gr.gwidth, gr.gheight
        self.starty, self.endy = gr.starty, gr.endy
        self.startx, self.endx = gr.startx, gr.endx

    def __del__(self):
        self.dev.close()

    def shell(self, s):
        #print(f'shell: {s}')
        self.dev.shell(s)

    def screen_dimensions(self):
        line = self.dev.shell('wm size')
        result = re.match(r"Physical size: (?P<height>\d+)x(?P<width>\d+)\n", line)
        return int(result.group('width')), int(result.group('height'))

    def input_swipe(self, p1, p2, time=100):
        start_pix = self.get_pixels(p1)
        end_pix = self.get_pixels(p2)

        self.shell(f'input swipe {start_pix[0]} {start_pix[1]} {end_pix[0]} {end_pix[1]} {time}')

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
        y,x = point

        dy = (self.endy - self.starty) / self.gheight
        dx = (self.endx - self.startx) / self.gwidth

        px = x * dx + dx/2 + self.startx
        py = y * dy + dy/2 + self.starty

        print(f"{x,y} becomes {px,py}")
        return round(px), round(py)


if __name__ == '__main__':
    gr = GridReader('screen.png')

    pairs = gr.read_dots()

    fs = FlowSolver(gr.gwidth, gr.gheight)
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
        gi = GridInputter(gr)
        for path in fs.paths:
            path = gi.simplify_path(path)
            print(fs.grid[path[0]])
            for i in range(len(path)-1):
                print(f"{path[i]} -> {path[i+1]}")
                gi.input_swipe( path[i], path[i+1] , time=200)
