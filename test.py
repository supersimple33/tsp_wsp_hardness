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

fig, ax = plt.subplots(1, 2, figsize=(18,9)) #12, 6

# points = file_load.load_points("data/pka379.tsp", False)
# points = file_load.load_points("data/custom3.tsp", False)
# points = util.generate_points(2) + util.circle_points(5, 4.0, ds.Point(15,-1))
# points = util.circle_points(5, 3.0, None) + util.circle_points(5, 4.0, ds.Point(15,-1)) + [ds.Point(20,20)]
# points = util.circle_points(5, 2.0, None) + util.circle_points(4, 2.0, ds.Point(15,0)) + util.circle_points(5, 2.0, ds.Point(15,15)) + util.circle_points(4, 2.0, ds.Point(0,15)) #+ [ds.Point(12,16)]
points = util.generate_points(16)

ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, points, ax, s=1.0)
fig.canvas.mpl_connect('button_press_event', lambda event: ts_problem.on_click(event, **DRAW_WSPD_ARGS)) # hook up interactions

start = time()
ts_problem.wspd
ts_problem.draw_wspd(**DRAW_WSPD_ARGS)
print("WSP took", time() - start)

print(len(ts_problem.quadtree))

# dynamic programming
start = time()
print(ts_problem.dp_path[1])
print("Test took", time() - start, end='\n')
ts_problem.draw_tour(ts_problem.dp_path[0], '#FFC0CB', label="DP")
assert ts_problem.check_tour(ts_problem.dp_path[0])

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
# start = time()
# print(ts_problem.nnn_path[1])
# print("Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.nnn_path[0], 'y', '-.', label="NNN")
# assert ts_problem.check_tour(ts_problem.nnn_path[0])

# smart nearest neighbor
start = time()
print(ts_problem.nwsp_path[1])
print("Test took", time() - start, end='\n')
ts_problem.draw_tour(ts_problem.nwsp_path[0], 'g', '--', label="NWSP")
assert ts_problem.check_tour(ts_problem.nwsp_path[0])

print("NN was off by:", ts_problem.nnn_path[1] / ts_problem.dp_path[1])
print("NWSP was off by:", ts_problem.nwsp_path[1] / ts_problem.dp_path[1])
ax[1].legend()

plt.show()

# tsp.print_wspd("points")


plt.show()
print()
