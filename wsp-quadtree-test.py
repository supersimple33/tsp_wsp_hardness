from wsp import wsp
from wsp import ds
from wsp import util
from wsp import cmd_parse
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# SPG = sub problem graph

# run algorithm
# >> python wsp-quadtree-test.py <points file> <separation factor> <quadtree:{-p/-point, -pr, -pmr, -pk/-pkpr}> <flags:{-d, -b<bucket_size>}>
# >> python wsp-quadtree-test.py att48.tsp 1 -p
# >> python wsp-quadtree-test.py att48.tsp 1 -pr -b4
# -d: debug info quadtree and WSP
# -bf: brute force (turns off WSP)

timeInit = time.perf_counter()
timeStart = time.perf_counter()

filename, s, wsp_mode, debug, shrink, quadtree, bucket = cmd_parse.parse_cmd(sys.argv)
# build WSP tree
wspTreeNode, wsp_count = wsp.runWSP(filename, s, debug, shrink, quadtree, bucket)

timeEnd = time.perf_counter()

# calculate well separated dictionary
points = wspTreeNode.get_points()

points = wspTreeNode.get_points()
num_points = len(points)

print("___________________________________________________________")
print(num_points, "points")
print(int(wsp_count), "WSPs found")
print(f"Loaded in {timeEnd - timeStart:0.4f} seconds")

#for i in range(len(minSolution) - 1):
    #wsp.ax[1].plot([minSolution[i].x, minSolution[i+1].x],[minSolution[i].y, minSolution[i+1].y], color="red")
wsp.ax[0].set_title(f"#WSP={wsp_count}, s={s}")
#wsp.ax[1].set_title(f"TSP Path: n={len(points)}, length={minDist:0.4f}")

print("___________________________________________________________")
plt.show()