import sys
import time

import matplotlib.pyplot as plt
from typeguard import check_type

from wsp import wsp
from wsp import ds
from wsp import util
from wsp import cmd_parse

nestable_points_t = ds.Point | list['nestable_points_t']


# run algorithm
# >> python tsp-spop.py <points file> <separation factor> <quadtree:{-pr, -point/-p}> <flags:{-d, -bf}>
# >> python tsp-spop.py att48.tsp 1 -pr
# -d: debug info quadtree and WSP
# -bf: brute force (turns off WSP)

diff_mult = 2 # REVIEW: what do you do?
timeInit = time.perf_counter()

filename, s, wsp_mode, debug, shrink, quadtree, bucket = cmd_parse.parse_cmd(sys.argv)
# build WSP tree
wspTreeNode, wsp_count = wsp.runWSP(filename, s, debug, shrink, quadtree, bucket)

timeStart = time.perf_counter()

# calculate well separated dictionary
ws = dict() # point -> set of well separated points (far away by WSP)
ws_orig = dict() # point a -> dict( WSP point b -> WSP set containing point a )
points = wspTreeNode.covered_points
for point in points:
    ws[point] = set()
    ws_orig[point] = dict()

points = wspTreeNode.covered_points
num_points = len(points)
print("___________________________________________________________")
print(num_points, "points")
print(int(wsp_count), "WSPs found")
print(f"Loaded in {timeStart - timeInit:0.4f} seconds")

# [ ([a,b,c],[d,e,f]) , ([a,b,c],[d,e,f]) ]
splits = [] # connects a notes covered points to every point those ones are well separated from

def find_relations(tree_node, add=True) -> set[ds.Point]: # REVIEW: what is the point of add param?
    """The consequences of this method revolve around building the splits"""
    sub_relations : set[ds.Point] = set()

    for node in tree_node.connection:
        sub_relations.update(node.covered_points) # TODO: check memory practices

    # recursively find relations
    if isinstance(tree_node, ds.AbstractPKQuadTree):
        for child in tree_node.children:
            sub_relations.union(find_relations(child, True)) # REVIEW: should we be utilizing unions?
    else:
        if tree_node.divided:
            sub_relations.union(find_relations(tree_node.ne, True))
            sub_relations.union(find_relations(tree_node.nw, True))
            sub_relations.union(find_relations(tree_node.sw, True))
            sub_relations.union(find_relations(tree_node.se, True))

    node_point_set = set(tree_node.covered_points)
    to_remove : list[ds.Point] = []
    if len(sub_relations) > 0:
        for p in sub_relations:
            if p in node_point_set:
                to_remove.append(p)
        for p in to_remove:
            sub_relations.remove(p)
        if add: # REVIEW: always true?
            if len(splits) == 0: # if no splits yet just add the pair in
                splits.append((node_point_set, sub_relations.copy()))
            else:
                for i, split in enumerate(splits):
                    # print(len(node_point_set), len(split[0]))
                    len_score1 = (len(node_point_set) + len(sub_relations)) / 2 - diff_mult * abs(len(node_point_set) - len(sub_relations))
                    len_score2 = (len(split[0]) + len(split[0])) / 2 - diff_mult * abs(len(split[0]) - len(split[0]))
                    if len_score1 > len_score2:
                        splits.insert(i, (node_point_set, sub_relations.copy()))
                        break

    # assert len(sub_relations) == 0
    return sub_relations

sr = find_relations(wspTreeNode, True)
print("splits", splits)

def apply_split(pair, glist) -> list['nestable_points_t']:
    """Applies a split to a jagged list structure"""
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
                # glist.remove(item)
            elif item in pair[1]:
                list2.append(item)
                # glist.remove(item)
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
# print(points)
print("grouped_points", grouped_points)

# traversal
# MARK: list utilities

def does_list_contain(point, lst):
    """Check if jagged list structure contains point"""
    return any(
        (isinstance(item, list) and does_list_contain(point, item)) # recursively search if subitem is a list
        or (not isinstance(item, list) and item == point) # if subitem is not list then check if it is the right point
        for item in lst # iterate over list items
    )

def get_points_from_list(lst):
    """Given jagged list returns a flattened list of the points"""
    return get_points_from_list_rec(lst, [])

def get_points_from_list_rec(lst, flattened_points):
    """Flattens points in jagged list into flattened_points"""
    for item in lst:
        if isinstance(item, list):
            get_points_from_list_rec(item, flattened_points)
        else:
            flattened_points.append(item)
    return flattened_points

def find_list_with(point, lst):
    """Returns the actual point from memory that is being stored in the list"""
    for item in lst:
        if isinstance(item, list):
            if does_list_contain(point, item):
                return item
        else:
            if item == point:
                return item
    return None

def clean_deep_lists(lst):
    """If lst is nesting a single element remove the nests"""
    if isinstance(lst, list) and len(lst) == 1:
        lst[0] = clean_deep_lists(lst[0])
        return lst[0]
    return lst


def find_path(start, glist, end, depth=-1, init=False):
    """Finds a path between start and end using the points from glist"""
    reverse = False
    if start is None:
        reverse = True
        start = end
        end = None
    glist = clean_deep_lists(glist)

    start_item = start
    end_item = end
    if init:
        start = None
        end = None
        end_item = None
    else:
        # if the start/end items are not at the surface of the glist then recursively search for them
        if start not in glist:
            start_item = find_list_with(start, glist)
        if end not in glist:
            end_item = find_list_with(end, glist)

    print("start", start, "end", end)
    print("start_item", start_item, "end_item", end_item)
    # print("glist", glist)

    rem : list[nestable_points_t] = glist.copy() # TODO: fixup rem typing further on
    if not init and ((isinstance(start_item, list) and end is not None and does_list_contain(end, start_item)) or (end_item == start_item)):
        # NOTE: this does not run on every tsp should probably REVIEW that
        # if start and end in same sublist, disassemble start sublist into rem
        print("BREAKING up dual connection subproblem")
        rem.remove(start_item)
        subitems = get_points_from_list(start_item)

        start_to_add = start if (start in subitems) else find_list_with(start, subitems)
        end_to_add = end if (end in subitems) else find_list_with(end, subitems)

        subitems.remove(start_to_add)
        subitems.remove(end_to_add)

        #print("starttoadd", start_to_add, "endtoadd", end_to_add, "subitems", subitems)
        if len(subitems) > 0:
            rem.append(subitems)
        rem.append(end_to_add)

        start_item = start_to_add
        end_item = end_to_add
    else:
        # regular remove behavior
        rem.remove(start_item)
    # print("rem", rem)

    # IMPORTANT
    trio_path = [(start, start_item, None)]
    # (entry point, the point block, exit point)

    # perform nearest neighbor on topmost layer
    def buildPerms(perm, remaining) -> list: # REVIEW: why is this an inner function?
        """Builds permutations of remaining points"""
        next_perms = []
        if len(remaining) == 0:
            return [perm]
        for r in remaining:
            last_point = perm[len(perm) - 1]
            new_point_list = perm.copy()
            new_point_list.append(r)
            new_rem = remaining.copy()
            new_rem.remove(r)
            if len(remaining) == 0:
                next_perms.append(new_point_list)
            else:
                next_perms.extend(buildPerms(new_point_list, new_rem))
        return next_perms

    # start at points[0]
    # print("Building perms!", len(rem))
    if len(rem) < 10: # REVIEW: what is rem in essence
        # TODO: make 10 a parameter
        print("BRUTE FORCE depth:", depth)
        perms = buildPerms([start_item], rem)
        # print("Perms:", len(perms))
        best_perm = None
        min_d = float('inf')
        for perm in perms:
            # if end != None and end == orig_start:
            #     perm.append(end)
            #     print("appended perm", perm)
            # print("Perm:",perm)
            # print("compare: ", start, perm[0], end, perm[len(perm) - 1], len(perms))
            # print("inlistL ", does_list_contain(orig_start, perm[0]), does_list_contain(end, perm[len(perm) - 1]))
            # print(does_list_contain(orig_start, perm[0]), does_list_contain(end, perm[len(perm) - 1]))
            test_perm = init
            if not test_perm:
                start_cond = False
                end_cond = False
                if isinstance(perm[0], list):
                    start_cond = does_list_contain(start, perm[0])
                else:
                    start_cond = start == perm[0]

                if isinstance(perm[len(perm) - 1], list):
                    end_cond = does_list_contain(end, perm[len(perm) - 1]) or end is None
                else:
                    end_cond = end == perm[len(perm) - 1] or end is None

                test_perm = start_cond and end_cond
            if init or test_perm:
                # print("not skip")
                dist = 0
                for i in range(len(perm) - 1):
                    p_A, p_B = util.min_proj_set_or_point(perm[i], perm[i+1])
                    dist += p_A.distance_to(p_B)
                # print(perm, dist)
                if dist < min_d:
                    min_d = dist
                    best_perm = perm
            # else:
            #     print("skipping")

        # best_perm
        # print("best perm - depth:", depth, min_d, best_perm)

        # trio_path = []
        for i in range(1,len(best_perm)):
            item = best_perm[i]
            prev_trio = trio_path[-1]
            prev_item = prev_trio[1]
            p_prev, p_cur, p2_prev, p2_cur = util.min_proj_set_or_point(prev_item, item, 2)
            if p_cur == end and i == len(best_perm) - 1:
                p_cur = p2_cur
            trio_path[-1] = (prev_trio[0], prev_trio[1], p_prev)
            item = clean_deep_lists(item)
            trio_path.append((p_cur, item, None))
    else:
        print("NEAREST NEIGHBOR depth:", depth, len(rem))
        while rem: # len > 0
            # Create a between blocks of remaining points noting the entry and exit point of the blocks
            
            prev_trio : (None | ds.Point, nestable_points_t, None) = trio_path[-1] # trio is (prev point, item, next point)
            prev_item = prev_trio[1] # find the last connection group
            minNext = None
            minNextDist = float('inf')
            for r in rem: # look through rem for next
                p_A : ds.Point
                p_B : ds.Point
                p_A, p_B = util.min_proj_set_or_point(prev_item, r)
                # NOTE: makes call specifically to util.min_proj since min2=False

                dist = p_A.distance_to(p_B)
                if dist < minNextDist and (len(rem) == 1 or r != end_item):
                    minNext = (p_A, r, p_B) # (prev point, next item, next point)
                    minNextDist = dist
            # print("minNext", minNext)
            trio_path[-1] = (prev_trio[0], prev_trio[1], minNext[0]) # (nestable_points_t, nestable_points_t, ds.Point)
            trio_path.append((minNext[2], minNext[1], None)) # (ds.Point, nestable_points_t, None)
            rem.remove(minNext[1])

    # tie the trio path together
    # assign end to end
    if init:
        # connect the first and last trio if possible # REVIEW: why only if possible? likely because we cant just settle Nones at the time of path construction # NOTE: this if rarely has any consequences
        first_trio = trio_path[0]
        last_trio = trio_path[-1]
        p_start, p_end, p_start2, p_end2 = util.min_proj_set_or_point(start_item, last_trio[1], 2)
        if first_trio[2] == p_start:
            p_start = p_start2
        if last_trio[0] == p_end:
            p_end = p_end2
        trio_path[0] = (p_start, first_trio[1], first_trio[2])
        trio_path[-1] = (last_trio[0], last_trio[1], p_end)
    elif end is not None:
        # connect the supplied end point from the outer into the trio path
        last_trio = trio_path[-1]
        trio_path[-1] = (last_trio[0], last_trio[1], end)

    # reroute trios with same start and end to second min projection
    for i in range(len(trio_path)-1):
        trio = trio_path[i]
        if trio[0] == trio[2]: # fix if same
            next_trio = trio_path[i + 1]
            _, _, p_A, _ = util.min_proj_set_or_point(trio[1], next_trio[1], True)
            if p_A is not None:
                # print("fixing trios", trio[0], trio[2], "=>", p_A)
                trio_path[i] = (trio[0], trio[1], p_A)
    for i, trio in enumerate(trio_path):
        if not isinstance(trio[1], list):
            trio_path[i] = (trio[1], trio[1], trio[1])

    print("trio path:", trio_path)
    # convert to list, make subcalls
    path = []
    last_trio = trio_path[-1]
    for trio in trio_path:
        if isinstance(trio[1], list):
            npath = find_path(trio[0], trio[1], trio[2], depth + 1)
            # print(npath, trio[0], trio[1], trio[2])
            path += npath
        else:
            path.append(trio[1])
    if reverse:
        path.reverse()
    return path


# search for permutations
rem_copy = grouped_points.copy()
# print(grouped_points[0])
solution = find_path(grouped_points[0], rem_copy, grouped_points[0], 0, True)
solution.append(solution[0])
# print(perm)

# find shortest permutation
minDist = util.calc_dist(solution)
timeEnd = time.perf_counter()

for index in range(len(solution) - 1):
    wsp.ax[1].plot([solution[index].x, solution[index+1].x],[solution[index].y, solution[index+1].y], color="red")
wsp.ax[0].set_title(f"#WSP={wsp_count}, s={s}")
wsp.ax[1].set_title(f"TSP Path: n={len(points)}, length={minDist:0.4f}")

print("")
print("Solution:", solution)
print("Solution Distance:", minDist)
# print(len(perms), "permutations examined")
print(f"Solution found in {timeEnd - timeStart:0.4f} seconds")
print("___________________________________________________________")
plt.show()

print()
