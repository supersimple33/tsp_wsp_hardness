from functools import cache, cached_property
from typing import Generic, TypeVar, Optional
from itertools import permutations, combinations

import matplotlib.pyplot as plt
from matplotlib import patches, cm
from matplotlib.axes import Axes
import numpy as np
from multimethod import multimethod

from wsp import ds
from wsp import util
from wsp.util import calc_dist, euclid_dist, group_by

BUFFER = 0.01 # TODO: this should prob just be a percentage of some sort

# QuadTreeType = TypeVar('QuadTreeType', bound=ds.AbstractQuadTree)
QuadTreeType = TypeVar('QuadTreeType', bound=ds.AbstractPKQuadTree)

class TravellingSalesmanProblem(Generic[QuadTreeType]): # TODO: better use of generics

    @multimethod
    def __init__(self, quadtree: ds.AbstractQuadTree, ax : np.ndarray[Optional[Axes]], s = 1.0, wspd = None): # TODO: utilize WSPD?
        """Initializes the TravellingSalesmanProblem, takes in an already built quadtree"""
        assert s >= 1.0, "Separation factor must be greater than or equal to 1.0"
        self._s = s

        self.quadtree = quadtree

        self.ax = ax
        if ax[0] is not None:
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMin], color="gray")
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMax, self.quadtree.boundary.yMax], color="gray")
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMin],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[0].plot([self.quadtree.boundary.xMax, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
        if ax[1] is not None:
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMin], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMax, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMin],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMax, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            
            self.current_screen = -1

        if isinstance(quadtree, ds.AbstractPKQuadTree) and not self.quadtree.pk_aggregated:
            self.quadtree.pk_aggregate(1) # REVIEW: 1 or 2 or more?
            self.quadtree.pk_draw()

        if not self.quadtree.path_compressed:
            self.quadtree = self.quadtree.path_compress() # TODO: Ensure these are inconsequential/not slow
            # REVIEW: is setter needed?
        self.quadtree.draw_points()

    @multimethod # wrenching multimethod is faster than overloading
    def __init__(self, treeType, points: list[ds.Point], ax : np.ndarray[Optional[Axes]], s = 1.0):
        """Initializes the TravellingSalesmanProblem, the base case version which builds its own quadtree"""
        assert s >= 1.0, "Separation factor must be greater than or equal to 1.0" # make stricter later?
        self._s = s

        # Setting/drawing the boundaries and initializing the quadtree
        width = max(p.x for p in points) - min(p.x for p in points)
        height = max(p.y for p in points) - min(p.y for p in points)
        minX = min(p.x for p in points) - (BUFFER * width)
        minY = min(p.y for p in points) - (BUFFER * height)
        maxX = max(p.x for p in points) + (BUFFER * width)
        maxY = max(p.y for p in points) + (BUFFER * height)

        boundary = ds.Rect(minX, minY, maxX, maxY)
        self.ax = ax

        self.quadtree : QuadTreeType = treeType(boundary, ax)

        if ax[0] is not None:
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMin], color="gray")
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMax, self.quadtree.boundary.yMax], color="gray")
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMin],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[0].plot([self.quadtree.boundary.xMax, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
        if ax[1] is not None:
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMin], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMax, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMin],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMax, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            
            self.current_screen = -1

        # Populating the quadtree and drawing the points
        for point in points:
            # self.quadtree.covered_points # REVIEW: can we remove this line?
            success = self.quadtree.insert(point)
            assert success

        if issubclass(treeType, ds.AbstractPKQuadTree):
            self.quadtree.pk_aggregate(1) # REVIEW: 1 or 2 or more?
            self.quadtree.pk_draw()

        self.quadtree = self.quadtree.path_compress()
        self.quadtree.draw_points()

    @property
    def points(self) -> list[ds.Point]:
        return self.quadtree.covered_points # REVIEW: is this slow????

    def draw_tour(self, tour: list[ds.Point], color='r', linestyle='-', label=None, linewidth=None):
        """Draws a path on the matplotlib axes"""
        if self.ax is None:
            print("No axes to draw on")
            return

        path = patches.PathPatch(patches.Path([point.to_tuple() for point in tour]), facecolor='none', edgecolor=color, linestyle=linestyle, label=label, linewidth=linewidth)
        self.ax[1].add_artist(path)

    # MARK: WSP

    @cached_property
    def wspd(self) -> set[tuple[frozenset[QuadTreeType, QuadTreeType], ds.SpecialDist]]:
        """Returns the well-seperated pair decomposition of the underlying quadtree, based on """
        ws_pairs = set()
        is_pk = issubclass(type(self.quadtree), ds.AbstractPKQuadTree)

        def recursive_wspd(node_A: QuadTreeType, node_B: QuadTreeType): # could be stricter with typing
            if len(node_A) == 0 or len(node_B) == 0 or (node_A.leaf and node_B.leaf and node_A == node_B):
                return

            big_radius = max(0 if node_A.leaf else node_A.radius, 0 if node_B.leaf else node_B.radius) # REVIEW: or just node_A.radius?
            small_radius = min(0 if node_A.leaf else node_A.radius, 0 if node_B.leaf else node_B.radius)

            # if node_A.points == [ds.Point(12.0, 17.0)] or node_B.points == [ds.Point(12.0, 17.0)]:
            #     if node_A.boundary.center().to_tuple() == (7.225, 16.700000000000003) or node_B.boundary.center().to_tuple() == (7.225, 16.700000000000003):
            #         pass

            # REVIEW: This property for Well Separation is subjective, is it the best option?
            # REVIEW: 2 * big_radius or big_radius * small_radius? REVIEW REVIEW REVIEW
            bubble_dist = (node_A.center - node_B.center).mag() - (big_radius + small_radius)
            if bubble_dist >= self._s * big_radius: # node_A guaranteed to be bigger
                special_dist = ds.SpecialDist.from_radius(bubble_dist, self._s * big_radius)
                ws_pairs.add((frozenset((node_A, node_B)), special_dist))
                return
            # else:
            #     pass # ??
            if (float("-inf") if node_A.leaf else node_A.radius) < (float("-inf") if node_B.leaf else node_B.radius): # REVIEW: correct comparator?
                # pull the most splittable node to the front
                node_A, node_B = node_B, node_A
            for child in node_A.children:
                recursive_wspd(child, node_B)

        recursive_wspd(self.quadtree, self.quadtree)

        return ws_pairs
    
    @cached_property
    def pair_sep_dict(self) -> dict[frozenset[QuadTreeType, QuadTreeType], ds.SpecialDist]:
        return {ab:c for ab,c in self.wspd}
    
    @cached_property
    def single_indexable_wspd(self) -> dict[ds.AbstractQuadTree, list[ds.AbstractQuadTree]]: # TODO: have this take over the old wspd property
        expanded_decomp = [(a,b) for (a,b), _ in self.wspd]
        expanded_decomp = expanded_decomp + [(b,a) for a,b in expanded_decomp]
        grouping = group_by(expanded_decomp, key=lambda x: x[0], value=lambda x: x[1])
        [g.sort(key=lambda b: self.pair_sep_dict[frozenset((a,b))]) for (a,g) in grouping.items()]
        return grouping


    # MARK: Drawing WSPs
    
    def draw_wsp_pair(self, node_A: QuadTreeType, node_B: QuadTreeType, no_leaves=False, use_boundary=False, no_circles=False, adjust=0.02, linewidth=1.0): # pretty sure the kwargs are safe to be extracted
        """Draws a single WSP pair on the matplotlib axes"""
        point_A = node_A.boundary.center() if use_boundary else (node_A.points[0] if node_A.leaf else node_A.center) # REVIEW: points[0] or mean point?
        point_B =  node_B.boundary.center() if use_boundary else (node_B.points[0] if node_B.leaf else node_B.center)
        midpoint = (point_A + point_B) / 2

        if no_leaves and node_A.leaf and node_B.leaf:
            return

        # draw the lines
        ls = '--' if node_A.leaf and node_B.leaf else '-'
        nudge = np.random.uniform(-1 * adjust * midpoint.mag(), adjust * midpoint.mag())
        color = np.random.rand(3)
        line = patches.PathPatch(patches.Path([(point_A.x, point_A.y),
                                                (midpoint.x + nudge, midpoint.y + nudge), # helpful in big picture mode to add curvature
                                                (point_B.x, point_B.y)]),
                                    linestyle=ls, linewidth=linewidth, zorder=5, facecolor='none', edgecolor=color)
        self.ax[0].add_patch(line)

        # draw the circles
        if not no_circles:
            big_radius = max(node_A.radius, node_B.radius) * (1 if node_A.leaf and node_B.leaf else 1)
            ls_A, ls_B = ':' if node_A.leaf else '-', ':' if node_B.leaf else '-'

            circle1 = patches.Circle((point_A.x, point_A.y), big_radius, color='r', fill=False, linestyle=ls_A)
            circle2 = patches.Circle((point_B.x, point_B.y), big_radius, color='r', fill=False, linestyle=ls_B)
            self.ax[0].add_artist(circle1) # calling add_artist instead of add_patch will cut off big circles
            self.ax[0].add_artist(circle2) # this is a god enough fix for now and circles still end up in patches

    def draw_wspd(self, no_leaves=False, use_boundary=False, no_circles=False, adjust=0.02, linewidth=1.0):
        """Draws the WSPD on the matplotlib axes by looping through each pair"""
        if self.ax is None:
            print("No axes to draw on")
            return
        self.ax[0].set_title(f"#WSP={len(self.wspd)}")
        # iterate through each wsp pair
        for (node_A, node_B), _ in self.wspd:
            self.draw_wsp_pair(node_A, node_B, no_leaves, use_boundary, no_circles, adjust, linewidth)

    def on_click(self, event, no_leaves=False, use_boundary=False, no_circles=False, adjust=0.02, linewidth=1.0): # TODO: extract args?
        """If left click cycle wsps if right click return to normal"""
        if len(self.wspd) == 0:
            print("No WSPs to show")
            return

        # clear all the patches
        for patch in self.ax[0].patches:
            patch.remove()
        if event.button == 1: # cycle
            self.current_screen = (self.current_screen + 1) % len(self.wspd)
            node_A, node_B, _ = self.wspd[self.current_screen]
            while no_leaves and node_A.leaf and node_B.leaf: # DANGER: infinite loop if only leaves
                self.current_screen = (self.current_screen + 1) % len(self.wspd)
                node_A, node_B, _ = self.wspd[self.current_screen]
            self.ax[0].set_title(f"Showing WSP#{self.current_screen + 1}/{len(self.wspd)}")
            self.draw_wsp_pair(node_A, node_B, no_leaves, use_boundary, no_circles, adjust, linewidth)
        elif event.button == 3: # display all
            self.draw_wspd(no_leaves, use_boundary, no_circles, adjust, linewidth)
        self.ax[0].figure.canvas.draw()

    def print_wspd(self, mode="center"):
        if mode == "center":
            for node_A, node_B, actual_s in self.wspd:
                print(node_A.center, node_B.center, actual_s)
        elif mode == "points":
            for node_A, node_B, actual_s in self.wspd:
                print(node_A.covered_points, node_B.covered_points, actual_s)

    # MARK: Paths

    @cached_property
    def untouched_path(self) -> tuple[list[ds.Point], float, tuple]:
        return self.points + (self.points[0],), calc_dist(self.points + (self.points[0],)), None

    @cached_property #@property # if timeit
    def brute_force_path(self) -> tuple[list[ds.Point], float, tuple]:
        """Returns the brute force path, set return_to_start to False for shortest path between
        all points not returning, used for nwsp"""
        min_solution = []
        min_dist = float('inf')

        perms = permutations(self.points[1:])
        num_perms = 0
        for perm in perms:
            num_perms += 1
            dist = euclid_dist(self.points[0], perm[0]) + calc_dist(perm) + euclid_dist(perm[-1], self.points[0])
            if dist < min_dist:
                min_solution = (self.points[0],) + perm + (self.points[0],)
                min_dist = dist

        return min_solution, min_dist, (num_perms,)

    @cached_property #@property # if timeit
    def dp_path(self) -> tuple[list[ds.Point], float, tuple]:
        """Returns a solution using dynamic programming based on held-karp"""
        n = len(self.points)
        arr = np.full((2**n, n), float('inf'))
        parent = np.full((2**n, n), -1) # record the parent node of each path so we can make later

        arr[0][0] = 0
        for i in range(1,n):
            arr[2 ** i][i] = euclid_dist(self.points[0], self.points[i])

        perms = 0
        for mask in range(2**n):
            for i in range(n): # TODO: check out that itertools.product thing
                if arr[mask][i] == float('inf'): # ensure that the path being travelled makes sense
                    continue
                for j in range(n):
                    perms += 1
                    if mask & (2 ** j) != 0: # ensure we are travelling to a new node
                        continue
                    new_mask = mask | (2 ** j)
                    if arr[new_mask][j] > arr[mask][i] + euclid_dist(self.points[i], self.points[j]):
                        arr[new_mask][j] = arr[mask][i] + euclid_dist(self.points[i], self.points[j])
                        parent[new_mask][j] = i

        path = [0]  # Start with the starting point
        mask = (2 ** n) - 1  # Initialize mask to represent all points

        # Rebuild the path
        while mask != 0:
            last_point = path[-1]
            next_point = parent[mask][last_point]
            path.append(next_point)
            mask ^= (1 << last_point)  # Remove the last point from the mask

        path[-1] = 0  # End with the starting point
        path = [self.points[e] for e in path]

        return path, calc_dist(path), (perms,)

    @cached_property
    def ishan_bfp_path(self) -> tuple[list[ds.Point], float, tuple]:
        """Ishan's implementation of using WSPs to prune brute force paths"""
        ws = dict() # point -> set of well separated points (far away by WSP)
        ws_orig = dict() # point a -> dict( WSP point b -> WSP set containing point a )
        points = self.quadtree.covered_points
        for p in points:
            ws[p] = set()
            ws_orig[p] = dict()

        queue = [self.quadtree]
        while len(queue) > 0:
            anode = queue.pop(0)
            for bnode in anode.connection:
                apoints = anode.covered_points
                bpoints = bnode.covered_points
                for a in apoints:
                    for b in bpoints:
                        # INVESIGATE: what are the consequences of this logic?
                        ws[a].add(b)
                        ws[b].add(a)
                        ws_orig[a][b] = anode
                        ws_orig[b][a] = bnode

            if issubclass(type(self.quadtree), ds.AbstractPKQuadTree):
                queue.extend(anode.children)
            else:
                if anode.divided:
                    queue.append(anode.ne)
                    queue.append(anode.nw)
                    queue.append(anode.sw)
                    queue.append(anode.se)

        def buildPerms(perm, rem):
            next_perms = []
            if len(perm) == len(self.points):
                return [perm]
            #print(len(rem))
            for r in rem:
                last_point = perm[len(perm) - 1]
                orig_set_finished = True
                if r in ws[last_point]: # checks if all points in last_point <-> r set have been visited
                    if r in ws_orig[last_point]:
                        for p in ws_orig[last_point][r]:
                            if p not in perm:
                                orig_set_finished = False
                if (r not in ws[last_point]) or (orig_set_finished):
                    new_point_list = perm.copy()
                    new_point_list.append(r)
                    new_rem = rem.copy()
                    new_rem.remove(r)
                    if len(new_point_list) == len(self.points):
                        next_perms.append(new_point_list)
                    else:
                        next_perms.extend(buildPerms(new_point_list, new_rem))
            return next_perms

        rem = points.copy()
        rem.remove(points[0])
        perms = buildPerms([points[0]], rem)

        min_solution = []
        min_dist = float('inf')
        for perm in perms:
            dist = calc_dist(perm) + euclid_dist(perm[-1], perm[0])
            if dist < min_dist:
                min_solution = perm + [perm[0]]
                min_dist = dist

        return min_solution, min_dist, (len(perms),)

    @cached_property
    def nnn_path(self) -> tuple[list[ds.Point], float, tuple]:
        """Returns a solution using the naive nearest neighbor"""
        path = [self.points[0]]
        rem = list(self.points[1:])
        while len(rem) > 0:
            min_dist = float('inf')
            min_point = None
            for p in rem: # TODO: make faster, refactor out
                dist = euclid_dist(path[-1], p)
                if dist < min_dist:
                    min_dist = dist
                    min_point = p
            path.append(min_point)
            rem.remove(min_point)
        path.append(self.points[0])
        return path, calc_dist(path), None

    def generate_sub_problem_order(self, start, t) -> list[QuadTreeType]:
        if start is None:
            start = max(self.single_indexable_wspd.keys(), key=len)

        collected_points : set[ds.Point] = set() # track which points we still need
        sub_problem_order : list[QuadTreeType] = [] # order in which to visit subproblems

        def try_to_add(subtree: QuadTreeType) -> bool:
            if subtree in sub_problem_order:
                return False
            elif any(point in collected_points for point in subtree.covered_points):
                return False # TODO: better handling
                # raise NotImplementedError("This should never happen")
            assert all(
                point not in collected_points for point in subtree.covered_points
            ), "This should never happen"
            if len(subtree) <= t:
                collected_points.update(subtree.covered_points)
                sub_problem_order.append(subtree)
                return True
            else:
                sub_tsp = TravellingSalesmanProblem(subtree, np.array([None, None]), self._s, None)
                sub_sub_order = sub_tsp.generate_sub_problem_order(None, t)
                sub_problem_order.extend(sub_sub_order)
                collected_points.update(subtree.covered_points) # REVIEW: Potentially dangerous
                return True
            return False


        current_subtree = start # node is a bad name for this
        assert try_to_add(current_subtree), "Should be guaranteed to insert the first one"
        while len(collected_points) < len(self.points):
            for problem in self.single_indexable_wspd[current_subtree]:
                if try_to_add(problem):
                    current_subtree = problem
                    break
            else:
                # REVIEW: what are we doing here why not just go up?
                jumpable = min(filter(lambda a: all(point not in collected_points for point in a.covered_points), self.single_indexable_wspd.keys()), key=lambda b: (b.center - current_subtree.center).mag())
                assert try_to_add(jumpable), "Should have selected a guaranteed jumpable node"
        return sub_problem_order

    @cache
    def nwsp_path(self, t=8) -> tuple[list[ds.Point], float, tuple]:
        biggest = max(self.single_indexable_wspd.keys(), key=len) # start with the subproblem with most points -> it should be the biggest
        # biggest = max(self.single_indexable_wspd[biggest], key=lambda x: x.radius) # choose the biggest radius ie what must be the most seperated
        # biggest = max(self.single_indexable_wspd.values(), key=lambda x: len(x[1]))[0] # choose the wsp with the most options
        
        sub_problem_order = self.generate_sub_problem_order(biggest, t=t)

        
        # MARK: Connect the subproblems
        tour : list[ds.Point] = []

        start, entry_point = util.min_proj(sub_problem_order[0].covered_points, sub_problem_order[1].covered_points)
        tour.append(start)
        for i in range(1, len(sub_problem_order) - 1):
            exit_point, next_entry = util.min_proj(sub_problem_order[i].covered_points, sub_problem_order[i + 1].covered_points)
            if sub_problem_order[i].leaf: # If we only have one point don't run bfp
                tour.append(sub_problem_order[i].covered_points[0])
                entry_point = next_entry
                continue
            elif entry_point == exit_point: # entry and exit may not be equal, TODO: make this better
                popped = tuple(filter(lambda x: x != entry_point, sub_problem_order[i].covered_points))
                exit_point, next_entry = util.min_proj(popped, sub_problem_order[i + 1].covered_points)
                # TODO: choose whichever new point is less costly
                # alt_prev, alt_entry = util.min_proj(sub_problem_order[i - 1].covered_points, popped)
                # alt_exit, alt_next_entry = util.min_proj(popped, sub_problem_order[i + 1].covered_points)
                # exit_added_cost = None
            tour.extend(util.hamiltonian_path(entry_point, exit_point, sub_problem_order[i].covered_points))
            entry_point = next_entry

        if sub_problem_order[-1].leaf and sub_problem_order[0].leaf:
            tour.extend((entry_point, start))
            return tour, calc_dist(tour), None # TODO: fill in None
        elif sub_problem_order[-1].leaf:
            tour.extend(util.hamiltonian_path(entry_point, start, sub_problem_order[0].covered_points + (entry_point,)))
            return tour, calc_dist(tour), None # TODO: fill in None
        elif sub_problem_order[0].leaf:
            tour.extend(util.hamiltonian_path(entry_point, start, sub_problem_order[-1].covered_points + (start,)))
            return tour, calc_dist(tour), None # TODO: fill in None

        exit_point, next_entry = util.min_proj(sub_problem_order[-1].covered_points, sub_problem_order[0].covered_points)
        if entry_point == exit_point or next_entry == start: # entry and exit may not be equal, TODO: make this better
            popped_end = tuple(filter(lambda x: x != entry_point, sub_problem_order[-1].covered_points)) if entry_point == exit_point else sub_problem_order[-1].covered_points
            popped_start = tuple(filter(lambda x: x != start, sub_problem_order[0].covered_points)) if next_entry == start else sub_problem_order[0].covered_points
            exit_point, next_entry = util.min_proj(popped_end, popped_start)
        tour.extend(util.hamiltonian_path(entry_point, exit_point, sub_problem_order[-1].covered_points))
        tour.extend(util.hamiltonian_path(next_entry, start, sub_problem_order[0].covered_points))

        return tour, calc_dist(tour), None # TODO: fill in None

    def draw_ordering(self, sub_problem_order: list[QuadTreeType]) -> None:
        """Draw the sub-tree ordering"""
        if self.ax is None:
            return
        n = len(sub_problem_order)
        color_map = cm.get_cmap('gist_rainbow', n)

        for i, node in enumerate(sub_problem_order):
            # color fill in the section under the current subproblem
            self.ax[1].fill_between([node.boundary.xMin, node.boundary.xMax], [node.boundary.yMin, node.boundary.yMin], [node.boundary.yMax, node.boundary.yMax], color=color_map(i), alpha=0.5)



    def spop_path(self) -> tuple[list[ds.Point], float, tuple]: # TODO: extract into new file
        pass

    # MARK: Testing

    def check_tour(self, path: list[ds.Point]) -> bool:
        """Checks if the path is valid for the tsp"""
        return (path[0] == path[-1] and (len(path) == len(self.points) + 1)
                and set(path) == set(self.points))

    def hk_lower_bound(self) -> tuple[list[ds.Point], float, tuple]:
        "Calculates the held karp lower bound based on Valenzuela's algorithm"
        raise NotImplementedError("Not yet implemented")
