import cv2
import numpy as np
import pycosat
from collections import defaultdict
import sys
import itertools as it
import time
from adb_shell.adb_device import AdbDeviceTcp
import re

class TentReader:
    def __init__(self, fname):
        self.img = cv2.imread(fname)
        self.sheight, self.swidth = self.img.shape[:2]
        self.gridlines_x = []
        self.gridlines_y = []

        self.grid_size = self.determine_grid_size()

        self.numlocs_rows = []
        self.numlocs_cols = []
        self.calculate_number_locations()

        self.centers = {}
        self.calc_centers()

        self.totals_rows = []
        self.totals_cols = []
        self.guess_numbers()

        self.trees = []
        self.read_grid()

    def count_dark_pixels_in(self, region):
        black = np.array([0,0,0], dtype='uint8')
        crop = self.img[
            region[0][1] : region[1][1],
            region[0][0] : region[1][0] ]
        return cv2.countNonZero(cv2.inRange(crop, black, black))

    def guess_numbers(self):
        TRANS = {
            114: 2,
            117: 3,
             66: 1,
            142: 0,
            136: 5,
            120: 4
        }
        self.totals_rows = []
        for i, rect in enumerate(self.numlocs_rows):
            numdark = self.count_dark_pixels_in(rect)
            self.totals_rows.append(TRANS[numdark])

        self.totals_cols = []
        for i, rect in enumerate(self.numlocs_cols):
            numdark = self.count_dark_pixels_in(rect)
            self.totals_cols.append(TRANS[numdark])

    def calc_centers(self):
        assert self.gridlines_x and self.gridlines_y
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.centers[x,y] = (
                    (self.gridlines_x[x] + self.gridlines_x[x+1])//2,
                    (self.gridlines_y[y] + self.gridlines_y[y+1])//2
                )
                cv2.circle(self.img, self.centers[x,y], 3, (255,0,0))

    def read_grid(self):
        BLANK = np.array((206,206,206), dtype='uint8')
        TREE  = np.array((  0,178,  0), dtype='uint8')
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                px,py = self.centers[x,y]
                k = self.img[py,px]
                print(x, y, self.img[py,px])
                if np.all(k == BLANK):
                    pass
                elif np.all(k == TREE):
                    self.trees.append((x,y))
                else:
                    raise ValueError()
        pass

    def __str__(self):
        s = ''
        for x in range(self.grid_size):
            s += str(self.totals_cols[x])
        s += '\n'
        s += '\n'
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x,y) in self.trees:
                    s += 'T'
                else:
                    s += '.'
            s += ' '
            s += str(self.totals_rows[y])
            s += '\n'
        return s

    def mark_number_locations(self):
        assert self.numlocs_rows and self.numlocs_cols
        for p1,p2 in self.numlocs_rows:
            cv2.rectangle(self.img, p1, p2, (255,0,0), 1)
        for p1,p2 in self.numlocs_cols:
            cv2.rectangle(self.img, p1, p2, (0,0,255), 1)

    def is_same_color(self, k, k2):
        if abs(k[0] - k2[0]) > 2: return False
        if abs(k[1] - k2[1]) > 2: return False
        if abs(k[2] - k2[2]) > 2: return False
        return True

    def calculate_number_locations(self):
        assert len(self.gridlines_x) == self.grid_size + 1
        assert self.is_arith_prog(self.gridlines_x)
        assert self.is_arith_prog(self.gridlines_y)

        dx = self.gridlines_x[-1] - self.gridlines_x[-2]
        dy = self.gridlines_y[-1] - self.gridlines_y[-2]

        self.numlocs_rows = [0] * self.grid_size
        self.numlocs_cols = [0] * self.grid_size

        px = self.gridlines_x[-1]
        for i in range(self.grid_size):
            self.numlocs_rows[i] = [
                (int(px+dx*0.70), int(self.gridlines_y[i]+dy*0.25)),
                (int(px+dx*1.10), int(self.gridlines_y[i]+dy*0.75))
            ]

        py = self.gridlines_y[-1]
        for i in range(self.grid_size):
            self.numlocs_cols[i] = [
                (int(self.gridlines_x[i]+dx*0.25), int(py+dy*0.50)),
                (int(self.gridlines_x[i]+dx*0.75), int(py+dy*1.10))
            ]

                
    def is_arith_prog(self, arr):
        arr = list(arr)
        return all(arr[i+2] - arr[i+1] == arr[i+1] - arr[i] for i in range(len(arr)-2))

    def extract_gridlines(self, arr):
        gridline_colors = [ [0,0,0], [49,49,49] ]
        gridlines = []
        in_gridline = True
        gridline_width = 0
        arr = list(arr)
        for i,k in enumerate(arr):
            if any(self.is_same_color(k, col) for col in gridline_colors):
                if not in_gridline:
                    # new gridline
                    in_gridline = True
                    gridline_width = 1
                else:
                    gridline_width += 1
            else:
                if in_gridline:
                    # gridline ended
                    in_gridline = False
                    # only count this if it was real
                    if 1 <= gridline_width < 3:
                        gridlines.append(i-gridline_width)
        return gridlines

    def determine_grid_size(self):
        cx = self.swidth // 2
        cy = self.sheight // 2
        print(f'{cx=}')
        print(f'{cy=}')

        # try to count the vertical lines
        gridline_width = 0
        in_gridline = False
        num_gridlines = 0
        self.gridlines_x = self.extract_gridlines(self.img[cy,px] for px in range(self.swidth))
        self.gridlines_y = self.extract_gridlines(self.img[py,cx] for py in range(self.sheight))
        print('gridlines_x', self.gridlines_x)
        print('gridlines_y', self.gridlines_y)
        return len(self.gridlines_x) - 1


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DIRECTIONS = (UP,DOWN,LEFT,RIGHT)
class TreeSolver:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.col_tots = [0] * grid_size
        self.row_tots = [0] * grid_size
        self.tree_locs = []

        # these are filled in with the solution
        self.tent_locs = []
        self.tree_dirs = {}

    def add_tree(self, x, y):
        self.tree_locs.append((x,y))

    def set_col_tot(self, x, tot):
        self.col_tots[x] = tot

    def set_row_tot(self, y, tot):
        self.row_tots[y] = tot

    def sat_encode_cell(self, x, y):
        val = 1 + y * self.grid_size + x
        assert val > 0
        return val

    def sat_encode_tree_dir(self, x, y, direc):
        # direc is one of [0,1,2,3]
        val = 1 + self.grid_size**2 + self.sat_encode_cell(x,y)*4 + direc
        assert val > 0
        return val

    def valid_space(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def ortho_neighbors(self, x, y):
        candidates = [
            (x+1,y),
            (x-1,y),
            (x,y+1),
            (x,y-1)
        ]
        return [(x,y) for (x,y) in candidates if self.valid_space(x,y)]

    def all_neighbors(self, x, y):
        candidates = [
            (x-1,y-1),
            (x-1,y),
            (x-1,y+1),

            (x,y-1),
            (x,y+1),

            (x+1,y-1),
            (x+1,y),
            (x+1,y+1)
        ]
        return [(x,y) for (x,y) in candidates if self.valid_space(x,y)]

    def translate(self, x, y, direc):
        if direc == UP:
            return (x, y-1)
        elif direc == DOWN:
            return (x, y+1)
        elif direc == LEFT:
            return (x-1, y)
        elif direc == RIGHT:
            return (x+1, y)

    def invert(self, direc):
        if direc == UP:
            return DOWN
        elif direc == DOWN:
            return UP
        elif direc == LEFT:
            return RIGHT
        elif direc == RIGHT:
            return LEFT

    def k_out_of_n_sat(self, vs, k):
        vs = list(vs)
        n = len(vs)
        # no more than k are true
        for combo in it.combinations(vs, k+1):
            # at least one of these is false
            yield [-c for c in combo]
        # at least k are true
        for combo in it.combinations(vs, n-k+1):
            # at least one of these is true
            yield [c for c in combo]

    def solve(self):
        constraints = []
        for (tx,ty) in self.tree_locs:
            # each tree points to a tent, so one of these cells has a tree
            adj_space_vars = [self.sat_encode_cell(nx,ny) for (nx,ny) in self.ortho_neighbors(tx,ty)]
            constraints.append(adj_space_vars)

            # each tree is pointing some direction
            constraints.append( [self.sat_encode_tree_dir(tx,ty,direc) for direc in DIRECTIONS] )
            # each tree is not pointing in two directions
            for d1,d2 in it.combinations(DIRECTIONS, 2):
                constraints.append( [-self.sat_encode_tree_dir(tx,ty,d1), -self.sat_encode_tree_dir(tx,ty,d2)] )

            # if a tree is pointing in a direction there's a tent there
            for d in DIRECTIONS:
                nx,ny = self.translate(tx,ty,d)
                if self.valid_space(nx,ny):
                    constraints.append( [-self.sat_encode_tree_dir(tx,ty,d), self.sat_encode_cell(nx,ny)] )

            # trees cannot be pointing outside the board
            if tx == 0:
                constraints.append([-self.sat_encode_tree_dir(tx,ty,LEFT)])
            if tx == self.grid_size-1:
                constraints.append([-self.sat_encode_tree_dir(tx,ty,RIGHT)])
            if ty == 0:
                constraints.append([-self.sat_encode_tree_dir(tx,ty,UP)])
            if ty == self.grid_size-1:
                constraints.append([-self.sat_encode_tree_dir(tx,ty,DOWN)])

            # no two trees can be pointing at the same square
            for d1 in DIRECTIONS:
                nx,ny = self.translate(tx,ty,d1)
                for d2 in DIRECTIONS:
                    nnx,nny = self.translate(nx,ny,d2)
                    if (nnx,nny) == (tx,ty): continue
                    if (nnx,nny) in self.tree_locs:
                        constraints.append([
                            -self.sat_encode_tree_dir(tx,ty,d1),
                            -self.sat_encode_tree_dir(nnx,nny,self.invert(d2))
                        ])
                        print(f'Added constraint {constraints[-1]} to prevent trees {tx,ty} and {nnx,nny} from sharing square {nx,ny}')

            # there is not a tent on a tree
            constraints.append([-self.sat_encode_cell(tx,ty)])
        
        # no two tents are adjacent
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for (nx,ny) in self.all_neighbors(x,y):
                    # these cannot both be tents
                    constraints.append([-self.sat_encode_cell(x,y), -self.sat_encode_cell(nx,ny)])

        # the "tree" is not pointing in any direction if there is no tree on a square
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x,y) not in self.tree_locs:
                    for d in DIRECTIONS:
                        constraints.append([-self.sat_encode_tree_dir(x,y,d)])

        # number of tents in each col must add up
        for x in range(self.grid_size):
            tot = self.col_tots[x]
            sat_vars = [self.sat_encode_cell(x,y) for y in range(self.grid_size)]
            constraints.extend( self.k_out_of_n_sat(sat_vars, tot) )

        # number of tents in each row must add up
        for y in range(self.grid_size):
            tot = self.row_tots[y]
            sat_vars = [self.sat_encode_cell(x,y) for x in range(self.grid_size)]
            constraints.extend( self.k_out_of_n_sat(sat_vars, tot) )

        print(f'encoded using {len(constraints)} clauses')

        soln = pycosat.solve(constraints)

        print(f'solution obtained: {soln}')

        bits = [0]*(max(abs(s) for s in soln)+1)
        for s in soln:
            if s > 0:
                bits[s] = 1


        # decode it into tent locations
        tent_locs = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                num = self.sat_encode_cell(x,y)
                if bits[num]:
                    tent_locs.append((x,y))
        self.tent_locs = tent_locs[:]
        
        for (x,y) in self.tree_locs:
            for d in DIRECTIONS:
                if bits[self.sat_encode_tree_dir(x,y,d)]:
                    self.tree_dirs[x,y] = d
        return tent_locs
    
    def __str__(self):
        s = ''
        for x in range(self.grid_size):
            s += str(self.col_tots[x])
        s += '\n'
        s += '\n'
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x,y) in self.tree_dirs:
                    d = self.tree_dirs[x,y]
                    s += {
                        UP:'^',
                        DOWN:'v',
                        LEFT:'<',
                        RIGHT:'>'
                    }[d]
                elif (x,y) in self.tree_dirs:
                    s += 'T'
                elif (x,y) in self.tent_locs:
                    s += '⌂'
                else:
                    s += '.'
            s += ' '
            s += str(self.row_tots[y])
            s += '\n'
        return s

                
class TentPlanter:
    def __init__(self, tentreader, host='localhost', port=5555):
        self.tr = tentreader
        self.dev = AdbDeviceTcp(host, port, default_transport_timeout_s=9.)
        self.dev.connect()
        self.swidth, sheight = self.screen_dimensions()
        self.grid_size = self.tr.grid_size

    def __del__(self):
        self.dev.close()

    def screen_dimensions(self):
        line = self.dev.shell('wm size')
        result = re.match(r"Physical size: (?P<height>\d+)x(?P<width>\d+)\n", line)
        return int(result.group('width')), int(result.group('height'))

    def input_tap(self, p):
        (px,py) = self.get_pixels(p)
        self.dev.shell(f'input tap {px} {py}')

    def get_pixels(self, p):
        return self.tr.centers[p]

if __name__ == '__main__':
    gr = TentReader(sys.argv[1])
    grid_size = gr.grid_size
    print('grid_size', grid_size)

    #cv2.imshow('img', gr.img)
    #ch = cv2.waitKey()
    #cv2.destroyAllWindows()

    print(gr)

    ts = TreeSolver(gr.grid_size)
    for (x,y) in gr.trees:
        ts.add_tree(x,y)

    for x in range(gr.grid_size):
        ts.set_col_tot(x, gr.totals_cols[x])
    for y in range(gr.grid_size):
        ts.set_row_tot(y, gr.totals_rows[y])

    tree_locs = ts.solve()
    print(tree_locs)
    print(ts)

    tp = TentPlanter(gr)
    for tree in tree_locs:
        print(f'tapping {tree}')
        tp.input_tap(tree)
        #time.sleep(1)
