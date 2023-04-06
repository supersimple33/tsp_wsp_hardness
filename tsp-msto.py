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
# >> python tsp-wgmst.py <points file> <separation factor> <quadtree:{-pr, -pmr}> <flags:{-d, -bf}>
# >> python tsp-wgmst.py att48.tsp 1 -pr
# -d: debug info quadtree and WSP
# -bf: brute force (turns off WSP)

timeInit = time.perf_counter()

filename, s, wsp_mode, debug, shrink, quadtree, bucket = cmd_parse.parse_cmd(sys.argv)
# build WSP tree
wspTreeNode, wsp_count = wsp.runWSP(filename, s, debug, shrink, quadtree, bucket)

timeStart = time.perf_counter()

# calculate well separated dictionary
points = wspTreeNode.get_points()

points = wspTreeNode.get_points()
num_points = len(points)

graph = {}
for p in points:
    graph[p] = set()

def build_wsp_graph(cur_node):
    if len(cur_node.connection) > 0:
        for node in cur_node.connection:
            p1, p2 = ds.min_proj(cur_node, node)
            graph[p1].add(p2)
            graph[p2].add(p1)

    if quadtree == ds.PKPRQuadTree or quadtree == ds.PKPMRQuadTree:
        for child in cur_node.children:
            build_wsp_graph(child)
    else:
        if cur_node.divided:
            build_wsp_graph(cur_node.ne)
            build_wsp_graph(cur_node.nw)
            build_wsp_graph(cur_node.sw)
            build_wsp_graph(cur_node.se)

build_wsp_graph(wspTreeNode)
print("graph:", graph)

edge_count_graph = set()
for p1 in graph:
    for p2 in graph[p1]:
        if (p1, p2) not in edge_count_graph:
            edge_count_graph.add((p1, p2))
            edge_count_graph.add((p2, p1))
            #wsp.ax[1].plot([p1.x, p2.x],[p1.y, p2.y], color="red")

print("___________________________________________________________")
print(num_points, "points")
print(int(wsp_count), "WSPs found")
print(f"Loaded in {timeStart - timeInit:0.4f} seconds")


queue = [(points[0], 0, None)]
added = set()
mst = {}
#mst_parent = {}

# Prims alg
while queue:
    # Choose the adjacent node with the least edge cost
    minCost = float('inf')
    minEdge = None
    for tup in queue:
        if tup[1] < minCost:
            minCost = tup[1]
            minEdge = tup
    queue.remove(minEdge)
    node = minEdge[0]
    prev = minEdge[2]
    #print(node)
    if prev != None and node not in added:
        if prev not in mst:
            mst[prev] = set()
        if node not in mst:
            mst[node] = set()
        mst[prev].add(node)
        mst[node].add(prev)
    #cost = queue[node]

    if node not in added:
        #min_span_tree_cost += cost
        added.add(node)
        #print("neighbor:", graph[node])
        for neighbor in graph[node]:
            if neighbor not in added:
                queue.append((neighbor, node.distance_to(neighbor), node))

print("mst:", mst)
drawn = set()
def draw_mst(node):
    #print(node)
    if node in mst and node not in drawn:
        drawn.add(node)
        for neighbor in mst[node]:
            if neighbor not in drawn:
                wsp.ax[1].plot([node.x, neighbor.x],[node.y, neighbor.y], color="orange")
                if neighbor in mst:
                    draw_mst(neighbor)
print(points[0])
draw_mst(points[0])

path = []
# optimize MST
def optimize_mst(node, lim=6):
    print("optimizing", node)
    # find collection of neighbor points
    to_optimize = set()

    def add_nodes(n):
        queue = [n]
        while len(queue) > 0:
            cur = queue[0]
            queue = queue[1:]
            '''for neighbor in mst[cur]:
                to_optimize.add(neighbor)
                if len(to_optimize) >= lim:
                    break
                queue.append(neighbor)
            if len(to_optimize) >= lim:
                    break'''

    add_nodes(node)
    print(to_optimize)



optimize_mst(points[0])

# traverse MST
'''def traverse_mst_for_path(prev, cur):
    # init prev to opposite of avg of neighbors
    if prev == None:
        avg = ds.Point(0,0)
        for child in mst[cur]:
            avg += child - cur
        avg /= len(mst[cur])
        prev = cur - avg

    vec_prev = (prev - cur).to_list()
    vec_prev_unit = vec_prev / np.linalg.norm(vec_prev)
    def calc_angle(node):
        vec_next = (node - cur).to_list()
        vec_next_unit = vec_next / np.linalg.norm(vec_next)
        dot_prod = np.dot(vec_prev_unit, vec_next_unit)
        angle = np.arccos(dot_prod)
        perp = [vec_next_unit[1], -vec_next_unit[0]]

        is_cc = np.dot(vec_prev_unit, perp) < 0
        # invert the angles for counter-clockwise rotations
        if is_cc:
            angle = 2*np.pi - angle

        return angle

    # order children
    ordered_children = []
    if cur in mst:
        for child in mst[cur]:
            angle = calc_angle(child)
            if len(ordered_children) == 0:
                ordered_children.append((child, angle))
            else:
                for i in range(len(ordered_children)):
                    if angle < ordered_children[i][1]:
                        ordered_children.insert(i, (child, angle))
                    if i == len(ordered_children)-1:
                        ordered_children.append((child, angle))
        # traverse in angle order
        for pair in ordered_children:
            traverse_mst_for_path(cur, pair[0])

    path.append(cur)

traverse_mst_for_path(None, points[0])'''

# find shortest permutation
minSolution = path
#minSolution.append(minSolution[0])
minDist = util.calcDist(path)

timeEnd = time.perf_counter()

#for i in range(len(minSolution) - 1):
    #wsp.ax[1].plot([minSolution[i].x, minSolution[i+1].x],[minSolution[i].y, minSolution[i+1].y], color="red")
wsp.ax[0].set_title(f"#WSP={wsp_count}, s={s}")
wsp.ax[1].set_title(f"TSP Path: n={len(points)}, length={minDist:0.4f}")

print("")
print("Solution:", minSolution)
print("Solution Distance:", minDist)
print(1, "permutations examined")
print(f"Solution found in {timeEnd - timeStart:0.4f} seconds")
print("___________________________________________________________")
plt.show()