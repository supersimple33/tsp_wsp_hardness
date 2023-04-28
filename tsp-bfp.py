from wsp import wsp
from wsp import ds
from wsp import util
from wsp import cmd_parse
from wsp import wsp_hardness
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Brute Force + WSP Pruning

# run algorithm
# >> python tsp-bfp.py <points file> <separation factor> <quadtree:{-point/-p, -pr, -pmr}> <flags:{-d, -bf}>
# >> python tsp-bfp.py custom1.tsp 1
# -d: debug info quadtree and WSP
# -bf: brute force (turns off WSP)

timeInit = time.perf_counter()

filename, s, wsp_mode, debug, shrink, quadtree, bucket = cmd_parse.parse_cmd(sys.argv)
# build WSP tree
wspTreeNode, wsp_count = wsp.runWSP(filename, s, debug, shrink, quadtree, bucket)

# calculate well separated dictionary
ws = dict() # point -> set of well separated points (far away by WSP)
ws_orig = dict() # point a -> dict( WSP point b -> WSP set containing point a )
points = wspTreeNode.get_points()
for p in points:
    ws[p] = set()
    ws_orig[p] = dict()

q = [wspTreeNode]
if wsp_mode:
    while len(q) > 0:
        anode = q[0]
        q = q[1:]
        # add WSP relationships to ws
        if len(anode.connection) > 0:
            for bnode in anode.connection:
                #wsp_count += 1
                #bnode = anode.connection
                apoints = anode.get_points()
                bpoints = bnode.get_points()
                for a in apoints:
                    for b in bpoints:
                        ws[a].add(b)
                        ws[b].add(a)
                        ws_orig[a][b] = apoints
                        ws_orig[b][a] = bpoints
        
        if quadtree == ds.PKPRQuadTree or quadtree == ds.PKPMRQuadTree:
            for child in anode.children:
                q.append(child)
        else:
            if anode.divided:
                q.append(anode.ne)
                q.append(anode.nw)
                q.append(anode.sw)
                q.append(anode.se)

points = wspTreeNode.get_points()
num_points = len(points)
print("___________________________________________________________")
print(num_points, "points")
print(int(wsp_count), "WSPs found")
#print(ws)

# traversal
solution = []
minSolution = []
minDist = float('inf')

def euclidDist(p1, p2):
    return math.sqrt( ((p2.x - p1.x) ** 2) + ((p2.y - p1.y) ** 2) )

def calcDist(points):
    dist = 0
    for i in range(len(points) - 1):
        dist += euclidDist(points[i], points[i+1])
    return dist

def buildPerms(perm, rem):
    next_perms = []
    if len(perm) == num_points:
        return [perm]
    #print(len(rem))
    for r in rem:
        last_point = perm[len(perm) - 1]
        orig_set_finished = True
        if r in ws[last_point]: # checks if all points in last_point <-> r set have been visited
            if (r in ws_orig[last_point]):
                for p in ws_orig[last_point][r]:
                    if (p not in perm):
                        orig_set_finished = False
        if (r not in ws[last_point]) or (orig_set_finished):
            new_point_list = perm.copy()
            new_point_list.append(r)
            new_rem = rem.copy()
            new_rem.remove(r)
            if len(new_point_list) == num_points:
                next_perms.append(new_point_list)
            else:
                next_perms.extend(buildPerms(new_point_list, new_rem))
    return next_perms

# start at points[0]
rem = points.copy()
rem.remove(points[0])
perms = buildPerms([points[0]], rem)

print(len(perms), "permutations examined")

for perm in perms:
    perm.append(perm[0])
    dist = calcDist(perm)
    if dist < minDist:
        minSolution = perm
        minDist = dist

#for i in range(len(minSolution) - 1):
#    wsp.ax[1].plot([minSolution[i].x, minSolution[i+1].x],[minSolution[i].y, minSolution[i+1].y], color="red")
#wsp.ax[0].set_title(f"#WSP={wsp_count}, s={s}")
#wsp.ax[1].set_title(f"TSP Path: n={len(points)}, length={minDist:0.4f}")

print("")
print("Solution:", minSolution)
print("Solution Distance:", minDist)
print("___________________________________________________________")
#plt.show()
#print(calcDist([wsp.Point(1,0),wsp.Point(8,1)]))
