from functools import cached_property
from typing import Generic, TypeVar, Type
from itertools import permutations, combinations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from wsp import ds
from wsp.util import calc_dist, euclid_dist

BUFFER = 1.1

TreeType = TypeVar('TreeType', bound=ds.AbstractQuadTree)

class TravellingSalesmanProblem(Generic[TreeType]):

    def __init__(self, treeType: Type[TreeType], points: list[ds.Point], ax : None | list[Axes], s = 1.0) -> None:
        """Initializes the TravellingSalesmanProblem"""

        assert s >= 1.0, "Separation factor must be greater than or equal to 1.0" # make stricter later?
        self._s = s

        # Setting/drawing the boundaries and initializing the quadtree
        minX = min(p.x for p in points) - BUFFER
        minY = min(p.y for p in points) - BUFFER
        maxX = max(p.x for p in points) + BUFFER
        maxY = max(p.y for p in points) + BUFFER

        boundary = ds.Rect(minX, minY, maxX, maxY)

        self.quadtree : TreeType = treeType(boundary, ax)

        self.ax = ax
        if ax is not None:
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMin], color="gray")
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMax, self.quadtree.boundary.yMax], color="gray")
            ax[0].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMin],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[0].plot([self.quadtree.boundary.xMax, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMin], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMax, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMin, self.quadtree.boundary.xMin],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")
            ax[1].plot([self.quadtree.boundary.xMax, self.quadtree.boundary.xMax],[self.quadtree.boundary.yMin, self.quadtree.boundary.yMax], color="gray")

        # Populating the quadtree and drawing the points
        for point in points:
            self.quadtree.covered_points
            success = self.quadtree.insert(point)
            assert success

        if issubclass(treeType, ds.AbstractPKQuadTree):
            self.quadtree.pk_aggregate(1) # REVIEW: 1 or 2 or more?
            self.quadtree.pk_draw()

        self.quadtree = self.quadtree.path_compress()
        self.quadtree.draw_points()

        self.splits = []
        self.points = points

    def draw_path(self, path: list[ds.Point], color='r', linestyle='-'):
        """Draws a path on the matplotlib axes"""
        if self.ax is None:
            print("No axes to draw on")
            return
        for i in range(len(path) - 1):
            self.ax[1].plot((path[i].x, path[i + 1].x), (path[i].y, path[i + 1].y), color=color, linestyle=linestyle)

    @cached_property
    def wspd(self) -> list[tuple[ds.AbstractQuadTree, ds.AbstractQuadTree]]:
        """Returns the well-seperated pair decomposition of the underlying quadtree, based on """
        ws_pairs = []
        is_pk = issubclass(type(self.quadtree), ds.AbstractPKQuadTree)

        def recursive_wspd(node_A: ds.AbstractQuadTree, node_B: ds.AbstractQuadTree): # could be stricter with typing
            if len(node_A) == 0 or len(node_B) == 0 or (node_A.leaf and node_B.leaf and node_A == node_B):
                return

            big_radius = max(0 if node_A.leaf else node_A.radius, 0 if node_B.leaf else node_B.radius) # REVIEW: or just node_A.radius?
            # REVIEW: This property for Well Separation is subjective, is it the best option?
            if (node_A.center - node_B.center).mag() - (big_radius + small_radius) >= self._s * big_radius: # node_A guaranteed to be bigger
                if (node_B, node_A) not in ws_pairs: # prevent dups
                    ws_pairs.append((node_A, node_B))
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

    def draw_wspd(self, no_leaves=False, use_boundary=False, no_circles=False, adjust=0.02, linewidth=1.0):
        if self.ax is None:
            print("No axes to draw on")
            return
        # iterate through each wsp pair
        for node_A, node_B in self.wspd:
            point_A = node_A.boundary.center() if use_boundary else (node_A.points[0] if node_A.leaf else node_A.center) # REVIEW: points[0] or mean point?
            point_B =  node_B.boundary.center() if use_boundary else (node_B.points[0] if node_B.leaf else node_B.center)
            midpoint = (point_A + point_B) / 2

            if no_leaves and node_A.leaf and node_B.leaf:
                continue

            # draw the lines
            ls = '--' if node_A.leaf and node_B.leaf else '-'
            nudge = np.random.uniform(-1 * adjust * midpoint.mag(), adjust * midpoint.mag())
            color = np.random.rand(3)
            self.ax[0].plot((point_A.x, midpoint.x + nudge), (point_A.y, midpoint.y + nudge), linestyle=ls, color=color, linewidth=linewidth)
            self.ax[0].plot((midpoint.x + nudge, point_B.x), (midpoint.y + nudge, point_B.y), linestyle=ls, color=color, linewidth=linewidth)

            # draw the circles
            if not no_circles:
                big_radius = max(node_A.radius, node_B.radius) * (0.5 if node_A.leaf and node_B.leaf else 1)
                ls_A, ls_B = ':' if node_A.leaf else '-', ':' if node_B.leaf else '-'

                circle1 = plt.Circle((point_A.x, point_A.y), big_radius, color='r', fill=False, linestyle=ls_A)
                circle2 = plt.Circle((point_B.x, point_B.y), big_radius, color='r', fill=False, linestyle=ls_B)
                self.ax[0].add_artist(circle1) # add_patch
                self.ax[0].add_artist(circle2)

    def print_wspd(self, mode="center"):
        if mode == "center":
            for node_A, node_B in self.wspd:
                print(node_A.center, node_B.center)
        elif mode == "points":
            for node_A, node_B in self.wspd:
                print(node_A.covered_points, node_B.covered_points)


    @cached_property #@property # if timeit
    def brute_force_path(self) -> tuple[list[ds.Point], tuple]:
        """Returns the brute force path"""
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

        return min_solution, (min_dist, num_perms)

    @cached_property #@property # if timeit
    def dp_path(self) -> tuple[list[ds.Point], tuple]:
        """Returns a solution using dynamic programming based on held-karp"""
        n = len(self.points)
        arr = np.full((2**n, n), float('inf'))
        parent = np.full((2**n, n), -1) # record the parent node of each path so we can make later

        arr[0][0] = 0
        for i in range(1,n):
            arr[2 ** i][i] = euclid_dist(self.points[0], self.points[i])

        for mask in range(2**n):
            for i in range(n):
                if arr[mask][i] == float('inf'): # ensure that the path being travelled makes sense
                    continue
                for j in range(n):
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

        return (path, (calc_dist(path), None))

    @cached_property
    def ishan_bfp_path(self) -> tuple[list[ds.Point], tuple]:
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

        return min_solution, (min_dist, len(perms))
