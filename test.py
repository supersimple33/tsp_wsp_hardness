from time import time

import matplotlib.pyplot as plt

from wsp import tsp
from wsp import util
from wsp import ds
from wsp import file_load

# QTREE = ds.PKPMRQuadTree
QTREE = ds.PKPRQuadTree
# QTREE = ds.PMRQuadTree
# QTREE = ds.PRQuadTree
# QTREE = ds.PointQuadTree

# points = points = util.generate_points(20)
# print(tsp.quadtree)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

# points = file_load.load_points("data/custom3.tsp", False)
points = util.generate_points(11)

tsp = tsp.TravellingSalesmanProblem[QTREE](QTREE, points, ax)

print(len(tsp.quadtree))

# # dynamic programming
start = time()
print(tsp.dp_path)
print("Test took", time() - start, end='\n')
tsp.draw_path(tsp.dp_path[0], 'b')

# brute force
start = time()
print(tsp.brute_force_path)
print("Test took", time() - start, " examind ", tsp.brute_force_path[1][1], " paths", end='\n')
tsp.draw_path(tsp.brute_force_path[0], 'r', '--')

# ish_bfp
start = time()
print(tsp.ishan_bfp_path)
print("Test took", time() - start, " examind ", tsp.ishan_bfp_path[1][1], " paths", end='\n')
tsp.draw_path(tsp.ishan_bfp_path[0], 'g', '-.')

print(tsp.dp_path[1][0]) 
print(tsp.brute_force_path[1][0])
print(tsp.ishan_bfp_path[1][0])

print()
