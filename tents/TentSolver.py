import cv2
import numpy as np
import sys

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

        self.grid = {}
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
                    self.grid[x,y] = '.'
                elif np.all(k == TREE):
                    self.grid[x,y] = 'T'
                else:
                    raise ValueError()
        pass

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


class TreeSolver:
    def __init__(self):
        pass

if __name__ == '__main__':
    gr = TentReader(sys.argv[1])
    grid_size = gr.grid_size
    print('grid_size', grid_size)

    cv2.imshow('img', gr.img)
    ch = cv2.waitKey()
    cv2.destroyAllWindows()
