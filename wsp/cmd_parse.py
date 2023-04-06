from wsp import ds
import sys


def parse_cmd(argv):
    filename = "data/custom1.txt"
    s = 1           # default separation factor
    wsp_mode = True # uses WSPs
    debug = False   # debug info for Quadtree and WSPs
    shrink = False  # shrink quadtree boundaries
    quadtree = ds.PMRQuadTree
    bucket = 1

    if (len(argv) >= 2):
        filename = "data/" + argv[1]
    if (len(argv) >= 3):
        s = float(argv[2])
    # check flags
    for arg in argv:
        if arg == "-pkpmr":
            quadtree = ds.PKPMRQuadTree
        if arg == "-pkpr":
            quadtree = ds.PKPRQuadTree
        if arg == "-pmr":
            quadtree = ds.PMRQuadTree
        if arg == "-pr":
            quadtree = ds.PRQuadTree
        if arg == "-point" or arg == "-p":
            quadtree = ds.PointQuadTree
        if arg[:2] == "-b":
            bucket = int(arg[2:])
        if arg == "-s":
            shrink = True
        if arg == "-d":
            debug = True
    
    return filename, s, wsp_mode, debug, shrink, quadtree, bucket

