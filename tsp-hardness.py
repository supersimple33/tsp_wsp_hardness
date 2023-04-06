# from wsp import wsp
from wsp import ds
from wsp import util
from wsp import cmd_parse
from wsp import wsp_hardness
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# run algorithm
# >> python tsp-nnp.py <points file> <separation factor> <quadtree:{-pr, -point/-p}> <flags:{-d, -bf}>
# >> python tsp-nnp.py att48.tsp 1 -p
# -d: debug info quadtree and WSP
# -bf: brute force (turns off WSP)

timeInit = time.perf_counter()

filename, s, wsp_mode, debug, shrink, quadtree, bucket = cmd_parse.parse_cmd(sys.argv)
# build WSP tree
# wspTreeNode, wsp_count = wsp.runWSP(filename, s, debug, shrink, quadtree, bucket)

wspTreeNode, wsp_count, kurtosis, tailed, avg_metric, var_metric, vals = wsp_hardness.hardness(filename, s, debug, shrink, quadtree, bucket)

# timeStart = time.perf_counter()

# # calculate well separated dictionary
# ws = dict() # point -> set of well separated points (far away by WSP)
# ws_orig = dict() # point a -> dict( WSP point b -> WSP set containing point a )
# points = wspTreeNode.get_points()
# for p in points:
#     ws[p] = set()
#     ws_orig[p] = dict()

# q = [wspTreeNode]
# if wsp_mode and False: #  
#     while len(q) > 0:
#         anode = q[0]
#         q = q[1:]
#         # add WSP relationships to ws
#         if len(anode.connection) > 0:
#             for bnode in anode.connection:
#                 #wsp_count += 1
#                 #bnode = anode.connection
#                 apoints = anode.get_points()
#                 bpoints = bnode.get_points()
#                 for a in apoints:
#                     for b in bpoints:
#                         ws[a].add(b)
#                         ws[b].add(a)
#                         ws_orig[a][b] = apoints
#                         ws_orig[b][a] = bpoints
#         if anode.divided:
#             q.append(anode.ne)
#             q.append(anode.nw)
#             q.append(anode.sw)
#             q.append(anode.se)
            
# points = wspTreeNode.get_points()
# num_points = len(points)
# print("___________________________________________________________")
# print(num_points, "points")
# print(int(wsp_count/2), "WSPs found")
# print(f"Loaded in {timeStart - timeInit:0.4f} seconds")

# print(points)

# # traversal
# solution = []
# minSolution = []
# minDist = float('inf')

# def findPath(start, rem):
#     perm = [start]
#     if len(perm) == num_points:
#         return [perm]
#     while len(rem) > 0:
#         minNext = None
#         minNextDist = float('inf')
#         for r in rem:
#             last_point = perm[len(perm) - 1]
#             orig_set_finished = True
#             if r in ws[last_point]: # checks if all points in last_point <-> r set have been visited
#                 print("blablabla\n")
#                 for p in perm:
#                     if (r in ws_orig[last_point]) and (p in ws_orig[last_point][r]):
#                         orig_set_finished = False
#             if (r not in ws[last_point]) or (orig_set_finished) or not wsp_mode:
#                 curNextDist = util.euclidDist(last_point, r)
#                 if curNextDist < minNextDist:
#                     minNext = r
#                     minNextDist = curNextDist
#         if minNext == None:
#             minNext = rem[0]
#         #print(minNext, rem)
#         perm.append(minNext)
#         rem.remove(minNext)
#     return perm

# # search for permutations
# perms = []
# for p in points:
#     rem = points.copy()
#     rem.remove(p)
#     path = findPath(p, rem)
#     path.append(path[0])
#     perms.append(path)
    
# # find shortest permutation
# for perm in perms:
#     dist = util.calcDist(perm)
#     if dist < minDist:
#         minSolution = perm
#         minDist = dist

# timeEnd = time.perf_counter()

# for i in range(len(minSolution) - 1):
#     wsp_hardness.ax[1].plot([minSolution[i].x, minSolution[i+1].x],[minSolution[i].y, minSolution[i+1].y], color="red")
wsp_hardness.ax[0].set_title(f"#WSP={wsp_count}, s={s}")
# wsp_hardness.ax[1].set_title(f"TSP Path: n={len(points)}, length={minDist:0.4f}")

# print("")
# print("Solution:", minSolution)
# print("Solution Distance:", minDist)
# print(len(perms), "permutations examined")
# print(f"Solution found in {timeEnd - timeStart:0.4f} seconds")
# print("___________________________________________________________")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.hist(vals, density=True, bins=30)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title(f"kurtosis: {kurtosis}, tailed: {tailed}, ")
plt.show()