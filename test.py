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

DRAW_WSPD_ARGS = {'no_leaves' : False, 'use_boundary' : True, 'no_circles' : False, 'adjust' : 0.0}

fig, ax = plt.subplots(1, 2, figsize=(12,6))

points = file_load.load_points("data/custom3.tsp", False)
# points = file_load.load_points("data/custom754.tsp", False)
# points = util.generate_points(40)

ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, points, ax, s=1.0)
fig.canvas.mpl_connect('button_press_event', lambda event: ts_problem.on_click(event, **DRAW_WSPD_ARGS)) # hook up interactions

print(len(ts_problem.quadtree))

# dynamic programming
start = time()
print(ts_problem.dp_path)
print("Test took", time() - start, end='\n')
ts_problem.draw_path(ts_problem.dp_path[0], '#FFC0CB')

# brute force
# start = time()
# print(tsp.brute_force_path)
# print("Test took", time() - start, " examind ", tsp.brute_force_path[1][1], " paths", end='\n')
# tsp.draw_path(tsp.brute_force_path[0], 'r', '--')

# ish_bfp
# start = time()
# print(tsp.ishan_bfp_path)
# print("Test took", time() - start, " examind ", tsp.ishan_bfp_path[1][1], " paths", end='\n')
# tsp.draw_path(tsp.ishan_bfp_path[0], 'g', '-.')

# naive nearest neighbor
start = time()
print(ts_problem.nnn_path)
print("Test took", time() - start, end='\n')
ts_problem.draw_path(ts_problem.nnn_path[0], 'y', '-.')

print(ts_problem.dp_path[1])
# print(tsp.brute_force_path[1])
# print(tsp.ishan_bfp_path[1])
print(ts_problem.nnn_path[1])

print("NN was off by:", ts_problem.nnn_path[1] / ts_problem.dp_path[1])

ts_problem.wspd
ts_problem.draw_wspd(**DRAW_WSPD_ARGS)
assert ts_problem.check_tour(ts_problem.dp_path[0])
assert ts_problem.check_tour(ts_problem.nnn_path[0])
# tsp.print_wspd("points")

plt.show()
print()
