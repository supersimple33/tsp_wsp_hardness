from time import time
import timeit
import random

import matplotlib.pyplot as plt
import numpy as np

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
THRESH : int = 2

fig, ax = plt.subplots(1, 2, figsize=(18,9)) #12, 6
# ax = np.array([None, None])

# MARK: - Labelling points
# fig, ax = plt.subplots(1, 1, figsize=(18,9)) #12, 6
# points = file_load.load_points("ALL_TSP/kroA100.tsp", False)
# ax.scatter([p.x for p in points], [p.y for p in points])
# for i in range(len(points)):
#     ax.annotate(i+1, (points[i].x, points[i].y))
# ax.axis("equal")

# points = file_load.load_points("ALL_TSP/a280.tsp", False)
# points = file_load.load_points("data/custom3.tsp", False)
# points = util.generate_points(2) + util.circle_points(5, 4.0, ds.Point(15,-1))
# points = util.circle_points(5, 3.0, None) + util.circle_points(5, 4.0, ds.Point(15,-1)) + [ds.Point(20,20)]
# points = util.circle_points(5, 2.0, None) + util.circle_points(4, 2.0, ds.Point(15,0)) + util.circle_points(5, 2.0, ds.Point(15,15)) + util.circle_points(4, 2.0, ds.Point(0,15)) #+ [ds.Point(12,16)]
points = util.generate_points(1089)
# points = [ds.Point(0,0), ds.Point(1,1), ds.Point(2,2), ds.Point(3,3)]


ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, points, ax, s=2.0)
fig.canvas.mpl_connect('button_press_event', lambda event: ts_problem.on_click(event, **DRAW_WSPD_ARGS)) # hook up interactions


# MARK: WSPD
# start = time()
# ts_problem.wspd
# print(len(ts_problem.wspd))
# print("WSP took", time() - start)
# ts_problem.draw_wspd(**DRAW_WSPD_ARGS)

x = timeit.timeit(lambda: print(len(ts_problem.quadtree)), number=1)
print(x)


x = timeit.timeit(lambda: ts_problem.dist_matrix)
print(x)

# MARK: branch and bound
# start = time()
# print(ts_problem.bab_path[1])
# print("BAB Test took", time() - start, end='\n')

# MARK: dynamic programming
# start = time()
# print(ts_problem.dp_path[1])
# print("DP Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.dp_path[0], '#000000', label="DP")
# assert ts_problem.check_tour(ts_problem.dp_path[0])

# MARK: Alt DP
# start = time()
# print(ts_problem.dp_alt_path[1])
# print("DP Alt Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.dp_alt_path[0], '#100000', label="DP Alt")
# assert ts_problem.check_tour(ts_problem.dp_alt_path[0])

# MARK: brute force
# start = time()
# print(ts_problem.brute_force_path)
# print("Test took", time() - start, " examind ", ts_problem.brute_force_path[1][1], " paths", end='\n')
# tsp.draw_path(ts_problem.brute_force_path[0], 'r', '--')

# MARK: ish_bfp
# start = time()
# print(tsp.ishan_bfp_path)
# print("Test took", time() - start, " examind ", tsp.ishan_bfp_path[1][1], " paths", end='\n')
# tsp.draw_path(tsp.ishan_bfp_path[0], 'g', '-.')

# MARK: naive nearest neighbor
start = time()
print(ts_problem.nnn_path[1])
print("Test took", time() - start, end='\n')
ts_problem.draw_tour(ts_problem.nnn_path[0], 'y', '-.', label="NNN")
assert ts_problem.check_tour(ts_problem.nnn_path[0])

# MARK: smart (or not so smart?) nearest neighbor
# start = time()
# print(ts_problem.nwsp_path(THRESH)[1])
# print("Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.nwsp_path(THRESH)[0], 'g', '--', label="NWSP")
# assert ts_problem.check_tour(ts_problem.nwsp_path(THRESH)[0])

# MARK: - Local Search
# start = time()
# print(ts_problem.local_search_path[1])
# print("LS Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.local_search_path[0], 'r', '-', label="LS")
# assert ts_problem.check_tour(ts_problem.local_search_path[0])

# MARK: - Quick Local Search
# start = time()
# print(ts_problem.quick_local_search_path[1])
# print("QLS Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.quick_local_search_path[0], '#A020F0', '-', label="QLS")
# assert ts_problem.check_tour(ts_problem.quick_local_search_path[0])

# MARK: - Simulated Annealing
# start = time()
# print(ts_problem.simulated_annealing_path[1])
# print("SA Test took", time() - start, end='\n')
# ts_problem.draw_tour(ts_problem.simulated_annealing_path[0], '#A020F0', '-', label="SA")
# assert ts_problem.check_tour(ts_problem.simulated_annealing_path[0])

# from python_tsp.heuristics import solve_tsp_simulated_annealing
# x = solve_tsp_simulated_annealing(ts_problem.dist_matrix, ts_problem.point_tour_to_ids(ts_problem.nnn_path[0][:-1]), alpha=0.9, rng=random.Random(ts_problem.number_seed))[0]
# assert x == ts_problem.point_tour_to_ids(ts_problem.simulated_annealing_path[0])[:-1]
# print("passed match")

# MARK: - LKH
start = time()
print(ts_problem.lkh_path[1])
print("LKH Test took", time() - start, end='\n')
ts_problem.draw_tour(ts_problem.lkh_path[0], '#FFC5CA', '-', label="LKH")
assert ts_problem.check_tour(ts_problem.lkh_path[0])
from python_tsp.heuristics import solve_tsp_lin_kernighan
assert ts_problem.point_tour_to_ids(ts_problem.lkh_path[0][:-1]) == solve_tsp_lin_kernighan(ts_problem.dist_matrix, ts_problem.point_tour_to_ids(ts_problem.nnn_path[0][:-1]))[0]


# print("NN was off by:", ts_problem.nnn_path[1] / ts_problem.dp_path[1])
# print("NWSP was off by:", ts_problem.nwsp_path(THRESH)[1] / ts_problem.dp_path[1])
ax[1].legend()

# ts_problem.draw_ordering(ts_problem.generate_sub_problem_order(None, THRESH))

plt.show()

# tsp.print_wspd("points")


plt.show()
print()
