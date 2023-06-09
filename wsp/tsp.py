from functools import cached_property
from typing import Generic, TypeVar, Type
from itertools import permutations

from matplotlib.axes import Axes
import numpy as np

from wsp import ds
from wsp.util import calc_dist, euclid_dist

BUFFER = 1.1

TreeType = TypeVar('TreeType', bound=ds.AbstractQuadTree)

class TravellingSalesmanProblem(Generic[TreeType]):

    def __init__(self, treeType: Type[TreeType], points: list[ds.Point], ax : None | list[Axes]) -> None:
        """Initializes the TravellingSalesmanProblem"""

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
            success = self.quadtree.insert(point)
            assert success

        if issubclass(treeType, ds.AbstractPKQuadTree):
            self.quadtree.pk_aggregate(2)
            self.quadtree.pk_draw()

        self.quadtree.draw_points()

        self.splits = []
        self.points = points

    def draw_path(self, path: list[ds.Point], color='r', linestyle='-'):
        """Draws a path on the matplotlib axes"""
        if self.ax is None:
            print()
            return
        for i in range(len(path) - 1):
            self.ax[1].plot((path[i].x, path[i + 1].x), (path[i].y, path[i + 1].y), color=color, linestyle=linestyle)

    # def run_wsp_setup(self):
    #     """Runs the WSP setup algorithm"""
    #     raise NotImplementedError

    @cached_property #@property # if timeit
    def brute_force_path(self) -> tuple[list[ds.Point], tuple]:
        """Returns the brute force path"""
        min_solution = []
        min_dist = float('inf')

        perms = permutations(self.points)
        num_perms = 0
        for perm in perms:
            num_perms += 1
            dist = calc_dist(perm) + euclid_dist(perm[-1], perm[0])
            if dist < min_dist:
                min_solution = perm + (perm[0],)
                min_dist = dist

        return min_solution, (min_dist, num_perms)

    @cached_property #@property # if timeit
    def dp_path(self) -> tuple[list[ds.Point], tuple]:
        """Returns a solution using dynamic programming based on held-karp"""
        n = len(self.points)
        arr = np.full((2**n, n), float('inf'))
        parent = np.full((2**n, n), -1) # record the parent node of each path so we can rebuild later

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

        while mask != 0:
            last_point = path[-1]
            next_point = parent[mask][last_point]
            path.append(next_point)
            mask ^= (1 << last_point)  # Remove the last point from the mask

        path[-1] = 0  # End with the starting point
        path = [self.points[e] for e in path]

        return (path, (calc_dist(path), None))
