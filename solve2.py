import cv2
import numpy as np

import matplotlib.colors
import webcolors
def get_colour_name(bgr_triplet):
    min_colours = {}
    for name, key in matplotlib.colors.CSS4_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - bgr_triplet[2]) ** 2
        gd = (g_c - bgr_triplet[1]) ** 2
        bd = (b_c - bgr_triplet[0]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def mask_of_color(im, color):
    mask = cv2.inRange(im, color, color)
    return mask

def myshow(im, message='im'):
    cv2.imshow(message, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_dot_centers(im):
    # strategy: look for the most prominent colors in the image
    histo = np.unique(im.reshape(-1, im.shape[-1]), axis=0, return_counts=True)
    print(f'{histo=}')
    arrindx = histo[1].argsort()
    colors = histo[0][arrindx[::-1]]
    
    # look for one with exactly two components of nearly the same size
    dot_size = 0

    for c in colors:
        mask = mask_of_color(im, c)
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, True, cv2.CV_32S)
        print(f'{numLabels=}')
        if numLabels == 3:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(numLabels)]
            print(f'{areas=}')
            if areas[2] <= areas[1] < areas[2] * 1.1 \
            or areas[1] <= areas[2] < areas[1] * 1.1:
                dot_size = (areas[1] + areas[2]) / 2
                print(f'{dot_size=}')
                #myshow(mask)
                break

    if not dot_size:
        raise Exception("Could not figure out dot size")

    dot_min = dot_size * 0.9
    dot_max = dot_size * 1.1

    for c in colors[1:]:
        name = get_colour_name(c)
        mask = mask_of_color(im, c)
        # open it up to avoid strangeness particularly around white
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        #myshow(mask)
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, True, cv2.CV_32S)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(numLabels)]
        print(f'{areas=}')
        if numLabels > 1 and sum(areas[1:]) < dot_min:
            break
        for i in range(1, numLabels):
            area = areas[i]
            centroid = (int(centroids[i][0]), int(centroids[i][1]))
            if dot_min < area < dot_max:
                # found a dot!
                print(f'dot! {name=} {area=} {centroid=}')
                cv2.circle(mask, centroid, 3, (0,0,255), 3)
                cv2.circle(mask, centroid, 20, (0,255,0), 3)
                #myshow(mask)
                yield centroid


def remove_dot(im, center):
    # just flood fill for now, we'll see if it works re anti aliasing
    color = im[center[1], center[0]]
    loDiff = (int(color[0])-1, int(color[1])-1, int(color[2])-1)
    loDiff = tuple(max(l,0) for l in loDiff)
    cv2.floodFill(im, None, center, (0,0,0), loDiff=loDiff, upDiff=(255,255,255), flags = 8 | cv2.FLOODFILL_FIXED_RANGE)


square_size = 0
wall_color = None

def find_neighbors(im, center):
    global square_size, wall_color
    red = (0,0,254)
    if center[0] < 0 or center[1] < 0:
        return
    if (im[center[::-1]] == red).all():
        # we've been here before
        return

    yield center

    # it is assumed that barriers are non-AA'd
    cv2.floodFill(im, None, center, red)
    #myshow(im)
    
    # start moving in each cardinal direction
    for direc in ( (0,1), (1,0), (0,-1), (-1,0) ):
        cur = center
        amt_traveled = 0
        try:
            while (im[cur[::-1]] == red).all() and (
            square_size == 0 or amt_traveled < square_size + 4):
                cur = (cur[0] + direc[0], cur[1] + direc[1])
                amt_traveled += 1
            if (im[cur[::-1]] == red).all():
                # didn't find shit
                continue
            if square_size == 0:
                square_size = amt_traveled * 2
            # we found a wall! maybe
            if wall_color is None:
                wall_color = im[cur[::-1]]
                print(f'{wall_color=} {get_colour_name(wall_color)}')
            if not (im[cur[::-1]] == wall_color).all():
                # not a real wall
                continue
            wall_width = 0
            while (im[cur[::-1]] == wall_color).all():
                cur = (cur[0] + direc[0], cur[1] + direc[1])
                wall_width += 1
            # move over a bit to ensure that we aren't right on the edge
            cur = (cur[0] + direc[0] * 10, cur[1] + direc[1] * 10)
            yield from find_neighbors(im, cur)
        except IndexError:
            continue
        except cv2.error:
            continue
    

def thresh_to_black(im):
    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(im2, (0,0,0), (180,255,35))
    myshow(mask, 'mask')
    imask = mask == 0
    out = np.zeros_like(im, np.uint8)
    out[imask] = im[imask]
    myshow(out, 'after thresh')
    return out

def get_cells(im, centers, color=(0,0,254)):
    # just connected components
    mask = mask_of_color(im, color)
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, True, cv2.CV_32S)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(numLabels)]
    # need to figure out what the most common area is... but it won't be exact
    sareas = sorted(areas)
    sareas = [a for a in sareas if a >= 100]
    print(f'{sareas=}')
    incs = [sareas[i+1] / sareas[i] for i in range(len(sareas)-1)]
    print(f'{incs=}')
    for start in range(len(incs)):
        if all(incs[start+i] < 1.05 for i in range(10)):
            # works for me
            break
    else:
        raise Exception("Couldn't find cells")
    end = start
    while incs[end] < 1.05:
        end += 1
    print(f'{start=}, {end=}')
    # okay so [start,end) is the region we care about
    indices = [i for i in range(len(areas)) if areas[i] in sareas[start:end]]
    
    for i in indices:
        yield (int(centroids[i][0]), int(centroids[i][1]))


def main(sfImage):
    im = cv2.imread(sfImage)
    im = thresh_to_black(im)
    centers = list(find_dot_centers(im))
    for center in centers:
        remove_dot(im, center)
        #myshow(im)
    cv2.imwrite('scratch/nodots.png', im)
    for center in centers:
        for neigh in find_neighbors(im, center):
            print(f'{neigh=}')
    myshow(im)
    for cell in get_cells(im, centers):
        print(f'{cell=}')
        cv2.floodFill(im, None, cell, (0,255,0))
    myshow(im)






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()

    main(args.image)
