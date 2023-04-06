from wsp import wsp
from wsp import ds
from wsp import util
from wsp import cmd_parse
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# run algorithm
# >> python tsp-sps.py <points file> <separation factor> <quadtree:{-pr, -point/-p}> <flags:{-d, -bf}>
# >> python tsp-sps.py att48.tsp 1 -pr
# -d: debug info quadtree and WSP
# -bf: brute force (turns off WSP)

timeInit = time.perf_counter()

filename, s, wsp_mode, debug, shrink, quadtree, bucket = cmd_parse.parse_cmd(sys.argv)
# build WSP tree
wspTreeNode, wsp_count = wsp.runWSP(filename, s, debug, shrink, quadtree, bucket)

timeStart = time.perf_counter()

# calculate well separated dictionary
ws = dict() # point -> set of well separated points (far away by WSP)
ws_orig = dict() # point a -> dict( WSP point b -> WSP set containing point a )
points = wspTreeNode.get_points()
for p in points:
    ws[p] = set()
    ws_orig[p] = dict()

points = wspTreeNode.get_points()
num_points = len(points)
print("___________________________________________________________")
print(num_points, "points")
print(int(wsp_count), "WSPs found")
print(f"Loaded in {timeStart - timeInit:0.4f} seconds")

# [ ([a,b,c],[d,e,f]) , ([a,b,c],[d,e,f]) ]
splits = []

def find_relations(tree_node, add=True):
    sub_relations = set()

    if len(tree_node.connection) > 0:
        for node in tree_node.connection:
            for p in node.get_points():
                sub_relations.add(p)

    if quadtree == ds.PKPRQuadTree or quadtree == ds.PKPMRQuadTree:
        for child in tree_node.children:
            sub_relations.union(find_relations(child, True))
    else:
        if tree_node.divided:
            sub_relations.union(find_relations(tree_node.ne, True))
            sub_relations.union(find_relations(tree_node.nw, True))
            sub_relations.union(find_relations(tree_node.sw, True))
            sub_relations.union(find_relations(tree_node.se, True))

    node_point_set = set(tree_node.get_points())
    to_remove = []
    if len(sub_relations) > 0:
        for p in sub_relations:
            if p in node_point_set:
                to_remove.append(p)
        for p in to_remove:
            sub_relations.remove(p)
        if add:
            #print(len(splits))
            if len(splits) == 0:
                splits.append((node_point_set, sub_relations.copy()))
            else:
                for i in range(len(splits)):
                    #print(len(node_point_set), len(splits[i][0]))
                    if len(node_point_set) > len(splits[i][0]):
                        splits.insert(i, (node_point_set, sub_relations.copy()))
                        break

            #splits.insert(0, (node_point_set, sub_relations.copy()))

    return sub_relations

find_relations(wspTreeNode, True)
#print(splits)

def apply_split(pair, glist):
    list1 = []
    list2 = []    
    to_remove = []
    to_add = []
    for item in glist:
        if isinstance(item, list):
            # recurse down list, add later
            to_remove.append(item)
            to_add.append(apply_split(pair, item.copy()))
        else:
            if item in pair[0]:
                list1.append(item)
                #glist.remove(item)
            elif item in pair[1]:
                list2.append(item)
                #glist.remove(item)
    for item in (list1 + list2 + to_remove):
        glist.remove(item)
    for item in to_add: # sublists
        glist.append(item)

    if len(list1) == 1:
        glist.append(list1[0])
    elif len(list1) > 0:
        glist.append(list1)
    if len(list2) == 1:
        glist.append(list2[0])
    elif len(list2) > 0:
        glist.append(list2)
    return glist

grouped_points = points.copy()

for pair in splits:
    if len(pair[0]) >= 2 or len(pair[1]) >= 2:
        grouped_points = apply_split(pair, grouped_points.copy())
#print(points)
#print("grouped_points", grouped_points)

# traversal

def does_list_contain(point, lst):
    for item in lst:
        if isinstance(item, list):
            if does_list_contain(point, item):
                return True
        else:
            if item == point:
                return True
    return False

def get_points_from_list(lst):
    return get_points_from_list_rec(lst, [])

def get_points_from_list_rec(lst, points):
    for item in lst:
        if isinstance(item, list):
            get_points_from_list_rec(item, points)
        else:
            points.append(item)
    return points

def find_list_with(point, lst):
    for item in lst:
        if isinstance(item, list):
            if does_list_contain(point, item):
                return item
        else:
            if item == point:
                return item
    return None

def clean_deep_lists(lst):
    if len(lst) == 1:
        lst[0] = clean_deep_lists(lst[0])
        return lst[0]
    return lst

def find_path(start, glist, end, depth=-1):
    loop = start == end # go back to start

    #print(start, end)
    reverse = False
    if start == None:
        reverse = True
        start = end
        end = None
        #rem = glist
    orig_start = start
    glist = clean_deep_lists(glist)

    #print("glist:", glist)
    #print("start:", orig_start, start)
    rem = glist.copy()
    if start not in glist:
        start = find_list_with(start, glist)

    if isinstance(start, list) and end != None and does_list_contain(end, start):
        # if start and end in same sublist, disassemble start sublist into rem
        rem.remove(start)
        rem += get_points_from_list(start)
        rem.remove(orig_start)
        start = orig_start
    else:    
        rem.remove(start) 

    if isinstance(orig_start, list):
        trio_path = [(None, start, None)]
    else:
        trio_path = [(orig_start, start, None)]
    if len(rem) == 0:
        #print("at the end, trio:",  start)
        trio_path = [(orig_start, start, None)]
    def point_list_contains(pointlist, point):
        if isinstance(pointlist, list):
            return does_list_contain(point, pointlist)
        else:
            return point == pointlist

    # perform nearest neighbor on topmost layer
    while len(rem) > 0:
        prev_trio = trio_path[len(trio_path) - 1] # trio is (prev point, item, next point)
        prev_item = prev_trio[1]
        minNext = None
        minNextDist = float('inf')
        for r in rem: # look through rem for next
            p_A, p_B = util.min_proj_set_or_point(prev_item, r)
            dist = p_A.distance_to(p_B)
            if dist < minNextDist and (len(rem) == 1 or not point_list_contains(r, end)):
                #if ((not r_is_list and r == end) or (r_is_list and does_list_contain(end, r))):
                #    print("new min:", r, len(rem), dist)
                minNext = (p_A, r, p_B) # (prev point, next item, next point)
                minNextDist = dist
        trio_path[len(trio_path) - 1] = (prev_trio[0], prev_trio[1], minNext[0])
        #print("set prev", trio_path[len(trio_path) - 1])
        trio_path.append((minNext[2], minNext[1], None))
        rem.remove(minNext[1])
        '''if len(rem) == 0 and loop:
            loop = False
            rem.append(end)'''

    # convert to list, make subcalls
    path = []
    #print("call info:", start, end)
    last_trio = trio_path[len(trio_path) - 1]
    trio_path[len(trio_path) - 1] = (last_trio[0], last_trio[1], end)
    #if depth < 2 or True:
    #    print(depth, "trio_path:", trio_path)
    for trio in trio_path:
        #print("trio:", trio)
        if isinstance(trio[1], list):
            npath = find_path(trio[0], trio[1], trio[2], depth + 1)
            #if depth < 2 or True:
            #    #print("calling", trio)
            #    print(depth + 1, "start and ends:", trio[0], trio[2])
            #    print(depth + 1, "res path:", reverse, npath)
            path += npath
        else:
            #print(depth + 1, "res point:", trio[1])
            path.append(trio[1])
    if reverse:
        path.reverse()
    #if loop:
        #path.append(path[0])
    return path


# search for permutations
perms = []
for item in grouped_points:
    rem = grouped_points.copy()
    perm = find_path(item, rem, item, 0)
    perm.append(perm[0])
    perms.append(perm)

# find shortest permutation
solution = []
minSolution = []
minDist = float('inf')
for perm in perms:
    dist = util.calcDist(perm)
    if dist < minDist:
        minSolution = perm
        minDist = dist
print("perms", perms)
timeEnd = time.perf_counter()

for i in range(len(minSolution) - 1):
    wsp.ax[1].plot([minSolution[i].x, minSolution[i+1].x],[minSolution[i].y, minSolution[i+1].y], color="red")
wsp.ax[0].set_title(f"#WSP={wsp_count}, s={s}")
wsp.ax[1].set_title(f"TSP Path: n={len(points)}, length={minDist:0.4f}")

print("")
print("Solution:", minSolution)
print("Solution Distance:", minDist)
print(len(perms), "permutations examined")
print(f"Solution found in {timeEnd - timeStart:0.4f} seconds")
print("___________________________________________________________")
plt.show()