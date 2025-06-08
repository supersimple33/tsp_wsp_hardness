from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from functools import cached_property

import numpy as np

from wsp import util


# @dataclass
class Point:
    """A point located at (x,y) in 2D space."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"P({self.x}, {self.y})"

    def __str__(self):
        return "P({:.4f}, {:.4f})".format(self.x, self.y)

    def __iter__(self):
        return iter(self.to_tuple())

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __add__(self, o):
        return Point(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Point(self.x - o.x, self.y - o.y)

    def __mul__(self, o):
        return Point(self.x * o, self.y * o)

    def __truediv__(self, o):
        return Point(self.x / o, self.y / o)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Point):
            return self.x == __value.x and self.y == __value.y
        elif isinstance(__value, (tuple, list)):
            return self.x == __value[0] and self.y == __value[1]
        else:
            return False

    def to_list(self):  # could make this lazy
        return [self.x, self.y]

    def to_tuple(self):  # could make this lazy
        return (self.x, self.y)

    def to_complex(self):
        return complex(self.x, self.y)

    def mag(self):
        return np.hypot(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

    @staticmethod
    def origin():
        return Point(0, 0)


@dataclass(order=True, frozen=True)
class SpecialDist:
    """This class adds support for ranking different distances between WSPs and tracks both the distance and the separation between the WSPs."""

    # sep: float
    dist: float
    sep: float

    def __repr__(self) -> str:
        return f"D({self.sep:.4f}, {self.dist:.4f})"

    @classmethod
    def from_radius(cls, dist, rel_radius):
        sep = float("inf") if rel_radius == 0 else dist / rel_radius
        # return cls(sep, dist)
        return cls(dist, sep)


class Rect:
    def __init__(self, xMin, yMin, xMax, yMax):
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax

    def __repr__(self) -> str:
        return str((self.xMin, self.yMin, self.xMax, self.yMax))

    def __str__(self) -> str:
        return "({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
            self.xMin, self.yMin, self.xMax, self.yMax
        )

    def contains(self, point: Point) -> bool:
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""
        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (
            point_x >= self.xMin
            and point_x < self.xMax
            and point_y >= self.yMin
            and point_y < self.yMax
        )

    def __contains__(self, point: Point | Tuple[int, int]) -> bool:
        return self.contains(point)

    def diameter(self) -> np.float32:  # TODO: convert to cached property
        # diagonal
        return np.hypot(self.xMax - self.xMin, self.yMax - self.yMin)

    def radius(self) -> np.float32:  # TODO: convert to cached property
        return self.diameter() / 2

    def center(self) -> Point:  # TODO: convert to cached property
        return Point((self.xMax + self.xMin) / 2, (self.yMax + self.yMin) / 2)


def min_dist(block_A, block_B):
    min_p1, min_p2 = min_proj(block_A, block_B)
    return min_p1.distance_to(min_p2)


def min_proj(block_A, block_B):
    """Min dist between points from Quadtree block_A and Quadtree block_B"""
    set_A = block_A.covered_points
    set_B = block_B.covered_points
    # print(set_A, set_B)
    return util.min_proj(set_A, set_B)  # min_p1, min_p2


def shrink_boundaries(block, regular=True):
    points = block.covered_points
    minX = float("inf")
    minY = float("inf")
    maxX = float("-inf")
    maxY = float("-inf")
    for p in points:
        if p.x < minX:
            minX = p.x
        if p.y < minY:
            minY = p.y
        if p.x > maxX:
            maxX = p.x
        if p.y > maxY:
            maxY = p.y
    minX -= 0.1
    minY -= 0.1
    maxX += 0.1
    maxY += 0.1
    # print(minX, minY, maxX, maxY)
    block.boundary = Rect(minX, minY, maxX, maxY)
    if regular:
        if block.divided:
            shrink_boundaries(block.nw)
            shrink_boundaries(block.ne)
            shrink_boundaries(block.se)
            shrink_boundaries(block.sw)
    else:
        for child in block.children:
            shrink_boundaries(child, False)
    return block


# Abstract QUADTREE


class AbstractQuadTree(ABC):
    """Abstract Quadtree implementation."""

    def __init__(self, boundary: Rect, ax, bucket=-1, depth=0):  # TODO: add parents
        """Initialize this node of the quadtree."""
        self.boundary = boundary  # boundary of current block
        self.bucket = bucket  # PointQT: doesn't use but setting here shouldnt matter
        self.ax = ax
        self.depth = depth  # mostly for string visualization spacing
        # self.points = [] # self.point = None
        self.connection: list[AbstractQuadTree] = []  # WSP connections
        self.divided = False  # flag for if divided into 4 child quads
        self.TreeType = type(self)
        self.leaf = False  # TODO: Implement this

        # self.repr = None # for wsp

        self.ne = None  # FIXME: these should be moved or should they?
        self.nw = None
        self.se = None
        self.sw = None

        assert bucket == 1, "non-singular buckets are not ready yet"

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of this node, suitably formatted."""
        pass

    # def __eq__(self, tree: object) -> bool:
    #     """Checks that two trees have the same structure and points (not just the same points)"""
    #     return self.points == tree.points and self.ne == tree.ne and self.nw == tree.nw and self.se == tree.se and self.sw == tree.sw

    def __hash__(self) -> int:  # REVIEW: if multiple calls this should be cached
        return hash(self.covered_points)

    @abstractmethod
    def str_short(self) -> str:  # TODO: default impl?
        pass

    @cached_property
    def diameter(self) -> np.float32:
        """Return the diameter of this node's boundary."""
        return self.boundary.diameter()

    @cached_property
    def radius(self) -> np.float32:
        """Return the radius of this node's boundary."""
        return self.boundary.radius()

    @cached_property
    def center(self) -> tuple:
        """Return the center of this node's boundary."""
        return self.boundary.center()

    def _clear_cache(self) -> None:
        if "_length" in self.__dict__:
            del self.__dict__[
                "_length"
            ]  # invalidate cached property since we added a point
        if "covered_points" in self.__dict__:
            del self.__dict__["covered_points"]

    @abstractmethod
    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point.
        This method has no consequences and only works to destroy cache's if the exist
        """
        self._clear_cache()  # NOTE: Best practice is to call the super to destroy cache

    @abstractmethod
    def insert(self, point) -> bool:
        """Insert the given point into this quadtree. This implementation is abstract. Nevertheless
        it is important that this super method is called so that the quadtree destroys its cache.
        """
        if point not in self.boundary:
            return False

        self._clear_cache()  # NOTE: again THIS SUPER METHOD MUST BE CALLED, so we can destroy

        # if self.repr is None or (self.repr - self.center).mag() > (point - self.center).mag():
        #     self.repr = point

        return True

    @abstractmethod
    def get_points_rec(self) -> list[Point]:
        """Get all points in this quadtree."""
        pass

    @cached_property
    def covered_points(self) -> tuple[Point]:
        """Get all points in this quadtree."""
        return self.get_points_rec()

    def __contains__(self, point: Point | Tuple[int, int]) -> bool:
        return point in self.covered_points

    @abstractmethod
    @cached_property
    def _length(self) -> int:
        """Return the number of points in this quadtree."""
        return 0

    def __len__(self):
        """Return the number of points in this quadtree."""
        return self._length

    def draw_points(self, color="black"):
        """Draw all points in this quadtree."""
        if self.ax[0] is None and self.ax[1] is None:
            return

        x = []
        y = []
        for p in self.covered_points:
            x.append(p.x)
            y.append(p.y)
        if self.ax[0] is not None:
            self.ax[0].scatter(x, y, color=color)
        if self.ax[1] is not None:
            self.ax[1].scatter(x, y, color=color)


class AbstractPKQuadTree(AbstractQuadTree):
    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        super().__init__(boundary, ax, bucket, depth)

        self.points = []  # center point
        self.children = []  # includes points and nodes

        self.pk_aggregated = False  # flag for if aggregated
        self.path_compressed = False  # flag for if path compressed
        self.leaf = False
        self.last_branch = False  # this is the last branch before a bunch of leaves

        self._radius = None  # Experimental

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        if self.pk_aggregated:
            sp = " " * self.depth * 2
            s = str(self.boundary) + " --> " + str(self.points)
            # print(self.depth, len(self.children))
            for c in self.children:
                s += "\n" + sp + "child:" + str(c)
            return s
        else:
            sp = " " * self.depth * 2
            s = str(self.boundary) + " --> " + str(self.points)
            if not self.divided:
                return s
            return (
                s
                + "\n"
                + "\n".join([sp + "se: " + str(self.se), sp + "sw: " + str(self.sw)])
            )

    def __eq__(self, tree: object) -> bool:
        """Checks that two trees have the same structure and points (not just the same points)"""
        if self.points != tree.points or len(self.children) != len(tree.children):
            return False
        if (
            self.ne != tree.ne
            or self.nw != tree.nw
            or self.se != tree.se
            or self.sw != tree.sw
        ):
            return False

        for child in self.children:
            if child not in tree.children:
                return False

        return True

    __hash__ = AbstractQuadTree.__hash__

    def str_short(self):
        return str(self.covered_points)  # str(self.boundary) +

    # MARK: comment this block out to return to default behavior
    @cached_property
    def diameter(self) -> np.float32:
        """Return the diameter of this node's boundary."""
        # return 0 if self.leaf else self.boundary.diameter()
        return 2 * self.radius

    @cached_property
    def radius(self) -> np.float32:
        """Return the radius of this node's boundary."""
        # return 0 if self.leaf else self.boundary.radius()
        return max(
            (p - self.center).mag() for p in self.covered_points
        )  # return the distance to the furthest point

    @cached_property
    def center(self) -> tuple:
        """Return the center of this node's boundary or the stored point if a leaf."""
        # return self.points[0] if self.leaf else self.boundary.center()
        return sum(self.covered_points, Point.origin()) / len(self.covered_points)

    def _clear_cache(self) -> None:
        super()._clear_cache()
        self._clear_pk_cache()

    def _clear_pk_cache(self) -> None:
        """Clear the cache to allow for new values now that self.leaf has been set."""
        if "diameter" in self.__dict__:
            del self.__dict__["diameter"]
        if "radius" in self.__dict__:
            del self.__dict__["radius"]
        if "center" in self.__dict__:
            del self.__dict__["center"]

    # @abstractmethod # NOTE: I would like to merge the divide methods due to their similarity but want to get some advice here first
    # def divide(self, subclass: Type['AbstractPKQuadTree']):
    #     """Divide (branch) this node by spawning four children nodes around a point."""
    #     super().divide()

    #     mid = Point(*self.boundary.center())
    #     self.nw = subclass(Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax), self.ax, self.bucket, self.depth+1)
    #     self.ne = subclass(Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax), self.ax, self.bucket, self.depth+1)
    #     self.se = subclass(Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y), self.ax, self.bucket, self.depth+1)
    #     self.sw = subclass(Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y), self.ax, self.bucket, self.depth+1)
    #     self.divided = True

    #     # # reinsert point, based on subclass # NOTE: I could just comment out and let each impl decide how to insert
    #     # points_to_reinsert = self.points
    #     # self.points = []
    #     # for p in points_to_reinsert: # NOTE: unsure of correct impl here can i use the first line or do i need the others
    #     #     # self.insert(p, True)
    #     #     self.ne.insert(p, True)
    #     #     self.nw.insert(p, True)
    #     #     self.se.insert(p, True)
    #     #     self.sw.insert(p, True)

    #     if self.ax is not None:
    #         self.ax[0].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
    #         self.ax[0].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")
    #         self.ax[1].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="lightgray")
    #         self.ax[1].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="lightgray")

    def get_points_rec(self) -> tuple[Point]:
        """Find the points in the quadtree that lie within boundary."""
        found_points = tuple(self.points)

        # if this node has children, search them too.
        assert (
            self.pk_aggregated
        ), "get_points_rec should only be called on aggregated trees"
        for child in self.children:
            found_points = found_points + child.covered_points

        return found_points

    def pk_aggregate(self, k, _parent=None):
        """Aggregate k-empty nodes into their grandparents. This should also path compress the tree."""
        # removes k-empty nodes and reassigns to grandparents
        self.pk_aggregated = True

        if self.divided:
            rec_children = []
            rec_children.append(self.nw.pk_aggregate(k, self))
            rec_children.append(self.ne.pk_aggregate(k, self))
            rec_children.append(self.se.pk_aggregate(k, self))
            rec_children.append(self.sw.pk_aggregate(k, self))
            for c in rec_children:
                if c is not None:
                    self.children.append(c)

            self.last_branch = all(child.leaf for child in self.children)

            # print(len(self.children), rec_children)
            if len(self.children) == 0:
                self.divided = False

            self.nw, self.ne, self.se, self.sw = None, None, None, None

            if _parent is not None:
                if len(self) < k:
                    # pass children upwards
                    # print("len", len(self), k)
                    _parent.children += self.children
                    return None
                else:
                    return self
        else:
            # print("leaf node", len(self.points))
            self.leaf = True
            self._clear_pk_cache()
            if len(self.points) > 0:
                assert len(self.points) == 1, "buckets are all size 1"
                return self
            else:
                return None

        return self

    def path_compress(self):
        self.path_compressed = True

        if self.leaf:  # if leaf node, return self
            return self
        if (
            len(self.children) == 1
        ):  # if only one child, prune itself by returning path compress of child
            return self.children[0].path_compress()
        else:
            for i, child in enumerate(self.children):
                self.children[i] = child.path_compress()
            return self

    def pk_draw(self):  # TODO: add none check
        if self.ax[1] is None:
            return

        if len(self) > 10000:
            print("are you sure you want to draw", len(self), "points?")

        for child in self.children:
            if len(child) > 1:
                child.pk_draw()
            # elif child.leaf:
            self.ax[1].plot(
                [child.boundary.xMin, child.boundary.xMax],
                [child.boundary.yMin, child.boundary.yMin],
                color="blue",
            )
            self.ax[1].plot(
                [child.boundary.xMin, child.boundary.xMax],
                [child.boundary.yMax, child.boundary.yMax],
                color="blue",
            )
            self.ax[1].plot(
                [child.boundary.xMin, child.boundary.xMin],
                [child.boundary.yMin, child.boundary.yMax],
                color="blue",
            )
            self.ax[1].plot(
                [child.boundary.xMax, child.boundary.xMax],
                [child.boundary.yMin, child.boundary.yMax],
                color="blue",
            )

    @cached_property
    def _length(self):
        """Return the number of points in the quadtree."""
        npoints = len(self.points)
        if (
            self.divided
        ):  # TODO: reevaluate this these checks may not be necessary based on aggregated status.
            npoints += len(self.nw) if self.nw is not None else 0
            npoints += len(self.ne) if self.ne is not None else 0
            npoints += len(self.se) if self.se is not None else 0
            npoints += len(self.sw) if self.sw is not None else 0
        for c in self.children:
            npoints += len(c)
        return npoints


# PK PR QUADTREE


class PKPMRQuadTree(AbstractPKQuadTree):
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        super().__init__(boundary, ax, bucket, depth)

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        super().divide()

        mid = Point(
            (self.boundary.xMin + self.boundary.xMax) / 2,
            (self.boundary.yMin + self.boundary.yMax) / 2,
        )
        self.nw = PKPMRQuadTree(
            Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.ne = PKPMRQuadTree(
            Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.se = PKPMRQuadTree(
            Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.sw = PKPMRQuadTree(
            Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
            # FIXME: I could use some clarification on whether the first line or following lines should be commented in or out
            # self.insert(p, True)

            self.ne.insert(p, True)
            self.nw.insert(p, True)
            self.se.insert(p, True)
            self.sw.insert(p, True)
        # draw
        if self.ax is not None:
            self.ax[0].plot(
                [mid.x, mid.x], [self.boundary.yMin, self.boundary.yMax], color="gray"
            )
            self.ax[0].plot(
                [self.boundary.xMin, self.boundary.xMax], [mid.y, mid.y], color="gray"
            )
            self.ax[1].plot(
                [mid.x, mid.x],
                [self.boundary.yMin, self.boundary.yMax],
                color="lightgray",
            )
            self.ax[1].plot(
                [self.boundary.xMin, self.boundary.xMax],
                [mid.y, mid.y],
                color="lightgray",
            )

    def insert(self, point, no_divide=False) -> bool:
        """Try to insert Point point into this QuadTree."""
        if not super().insert(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
            self.points.append(point)
            if not no_divide and len(self.points) > 1:
                self.divide()

            return True

        return (
            self.ne.insert(point)
            or self.nw.insert(point)
            or self.se.insert(point)
            or self.sw.insert(point)
        )


# PK PR QUADTREE


class PKPRQuadTree(AbstractPKQuadTree):
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        super().__init__(boundary, ax, bucket, depth)

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        super().divide()

        mid = Point(
            (self.boundary.xMin + self.boundary.xMax) / 2,
            (self.boundary.yMin + self.boundary.yMax) / 2,
        )
        self.nw = self.TreeType(
            Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.ne = self.TreeType(
            Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.se = self.TreeType(
            Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.sw = self.TreeType(
            Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
            self.insert(p)
        # draw
        if self.ax[0] is not None:
            self.ax[0].plot(
                [mid.x, mid.x], [self.boundary.yMin, self.boundary.yMax], color="gray"
            )
            self.ax[0].plot(
                [self.boundary.xMin, self.boundary.xMax], [mid.y, mid.y], color="gray"
            )
        if self.ax[1] is not None:
            self.ax[1].plot(
                [mid.x, mid.x],
                [self.boundary.yMin, self.boundary.yMax],
                color="lightgray",
            )
            self.ax[1].plot(
                [self.boundary.xMin, self.boundary.xMax],
                [mid.y, mid.y],
                color="lightgray",
            )

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if not super().insert(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
            if len(self.points) < self.bucket:
                # Node doesn't have a point yet.
                self.points.append(point)
                return True

            # Already leaf: divide if necessary, then try the sub-quads.
            self.divide()

        return (
            self.ne.insert(point)
            or self.nw.insert(point)
            or self.se.insert(point)
            or self.sw.insert(point)
        )


# PMR QUADTREE


class PMRQuadTree(AbstractQuadTree):
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=-1, depth=0):
        """Initialize this node of the quadtree."""
        super().__init__(boundary, ax, bucket, depth)
        self.points = []  # center point

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = " " * self.depth * 2
        s = str(self.boundary) + " --> " + str(self.points)
        if not self.divided:
            return s
        return (
            s
            + "\n"
            + "\n".join(
                [
                    sp + "nw: " + str(self.nw),
                    sp + "ne: " + str(self.ne),
                    sp + "se: " + str(self.se),
                    sp + "sw: " + str(self.sw),
                ]
            )
        )

    def str_short(self):
        return str(self.boundary)

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        super().divide()

        mid = Point(
            (self.boundary.xMin + self.boundary.xMax) / 2,
            (self.boundary.yMin + self.boundary.yMax) / 2,
        )
        self.nw = PMRQuadTree(
            Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.ne = PMRQuadTree(
            Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.se = PMRQuadTree(
            Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.sw = PMRQuadTree(
            Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
            # self.insert(p, True)
            self.ne.insert(p, True)
            self.nw.insert(p, True)
            self.se.insert(p, True)
            self.sw.insert(p, True)
        # draw
        if self.ax is not None:
            self.ax[0].plot(
                [mid.x, mid.x], [self.boundary.yMin, self.boundary.yMax], color="gray"
            )
            self.ax[0].plot(
                [self.boundary.xMin, self.boundary.xMax], [mid.y, mid.y], color="gray"
            )
            self.ax[1].plot(
                [mid.x, mid.x], [self.boundary.yMin, self.boundary.yMax], color="gray"
            )
            self.ax[1].plot(
                [self.boundary.xMin, self.boundary.xMax], [mid.y, mid.y], color="gray"
            )

    def insert(self, point, no_divide=False):
        """Try to insert Point point into this QuadTree."""
        if not super().insert(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
            self.points.append(point)
            if not no_divide and len(self.points) > 1:
                self.divide()

            return True

        return (
            self.ne.insert(point)
            or self.nw.insert(point)
            or self.se.insert(point)
            or self.sw.insert(point)
        )

    def get_points_rec(self):
        """Find the points in the quadtree that lie within boundary."""
        found_points = tuple(self.points)

        # if this node has children, search them too.
        if self.divided:
            found_points = (
                found_points
                + self.nw.covered_points
                + self.ne.covered_points
                + self.se.covered_points
                + self.sw.covered_points
            )
        return found_points

    @cached_property
    def _length(self):
        """Return the number of points in the quadtree."""
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw) + len(self.ne) + len(self.se) + len(self.sw)
        return npoints


# POINT REGION QUADTREE
class PRQuadTree(AbstractQuadTree):
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        super().__init__(boundary, ax, bucket, depth)
        self.points = []  # center point

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = " " * self.depth * 2
        s = str(self.boundary) + " --> " + str(self.points)
        if not self.divided:
            return s
        return (
            s
            + "\n"
            + "\n".join(
                [
                    sp + "nw: " + str(self.nw),
                    sp + "ne: " + str(self.ne),
                    sp + "se: " + str(self.se),
                    sp + "sw: " + str(self.sw),
                ]
            )
        )

    def str_short(self):
        return str(self.covered_points)  # str(self.boundary) +

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        super().divide()

        mid = Point(
            (self.boundary.xMin + self.boundary.xMax) / 2,
            (self.boundary.yMin + self.boundary.yMax) / 2,
        )
        self.nw = self.TreeType(
            Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.ne = self.TreeType(
            Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.se = self.TreeType(
            Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.sw = self.TreeType(
            Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y),
            self.ax,
            self.bucket,
            self.depth + 1,
        )
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
            self.insert(p)
        # draw
        if self.ax is not None:
            self.ax[0].plot(
                [mid.x, mid.x], [self.boundary.yMin, self.boundary.yMax], color="gray"
            )
            self.ax[0].plot(
                [self.boundary.xMin, self.boundary.xMax], [mid.y, mid.y], color="gray"
            )
            self.ax[1].plot(
                [mid.x, mid.x], [self.boundary.yMin, self.boundary.yMax], color="gray"
            )
            self.ax[1].plot(
                [self.boundary.xMin, self.boundary.xMax], [mid.y, mid.y], color="gray"
            )

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if not super().insert(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
            if len(self.points) < self.bucket:
                # Node doesn't have a point yet.
                self.points.append(point)
                return True

            # Already leaf: divide if necessary, then try the sub-quads.
            self.divide()

        return (
            self.ne.insert(point)
            or self.nw.insert(point)
            or self.se.insert(point)
            or self.sw.insert(point)
        )

    def get_points_rec(self):
        """Find the points in the quadtree that lie within boundary."""
        found_points = tuple(self.points)

        # if this node has children, search them too.
        if self.divided:
            found_points = (
                found_points
                + self.nw.covered_points
                + self.ne.covered_points
                + self.se.covered_points
                + self.sw.covered_points
            )
        return found_points

    @cached_property
    def _length(self):
        """Return the number of points in the quadtree."""
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw) + len(self.ne) + len(self.se) + len(self.sw)
        return npoints


# POINT QUADTREE ##############################################################
class PointQuadTree(AbstractQuadTree):
    """Point Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=-1, depth=0):
        """Initialize this node of the quadtree."""
        super().__init__(boundary, ax, bucket, depth)
        self.point = None  # center point

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = " " * self.depth * 2
        s = str(self.boundary) + " --> " + str(self.point)
        if not self.divided:
            return s
        return (
            s
            + "\n"
            + "\n".join(
                [
                    sp + "nw: " + str(self.nw),
                    sp + "ne: " + str(self.ne),
                    sp + "se: " + str(self.se),
                    sp + "sw: " + str(self.sw),
                ]
            )
        )

    def str_short(self):
        return str(self.boundary)

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        super().divide()

        self.nw = PointQuadTree(
            Rect(self.boundary.xMin, self.point.y, self.point.x, self.boundary.yMax),
            self.ax,
            self.depth + 1,
        )
        self.ne = PointQuadTree(
            Rect(self.point.x, self.point.y, self.boundary.xMax, self.boundary.yMax),
            self.ax,
            self.depth + 1,
        )
        self.se = PointQuadTree(
            Rect(self.point.x, self.boundary.yMin, self.boundary.xMax, self.point.y),
            self.ax,
            self.depth + 1,
        )
        self.sw = PointQuadTree(
            Rect(self.boundary.xMin, self.boundary.yMin, self.point.x, self.point.y),
            self.ax,
            self.depth + 1,
        )
        self.divided = True
        # draw
        if self.ax is not None:
            self.ax[0].plot(
                [self.point.x, self.point.x],
                [self.boundary.yMin, self.boundary.yMax],
                color="gray",
            )
            self.ax[0].plot(
                [self.boundary.xMin, self.boundary.xMax],
                [self.point.y, self.point.y],
                color="gray",
            )
            self.ax[1].plot(
                [self.point.x, self.point.x],
                [self.boundary.yMin, self.boundary.yMax],
                color="gray",
            )
            self.ax[1].plot(
                [self.boundary.xMin, self.boundary.xMax],
                [self.point.y, self.point.y],
                color="gray",
            )

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if not super().insert(point):
            # The point does not lie inside boundary: bail.
            return False

        if self.point == None:
            # Node doesn't have a point yet.
            self.point = point
            return True

        # Already leaf: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (
            self.ne.insert(point)
            or self.nw.insert(point)
            or self.se.insert(point)
            or self.sw.insert(point)
        )

    def get_points_rec(self):
        """Find the points in the quadtree that lie within boundary."""
        found_points = tuple(self.points)

        # if this node has children, search them too.
        if self.divided:
            found_points = (
                found_points
                + self.nw.covered_points
                + self.ne.covered_points
                + self.se.covered_points
                + self.sw.covered_points
            )
        return found_points

    @cached_property
    def _length(self):
        """Return the number of points in the quadtree."""
        if self.point != None:
            npoints = 1
        else:
            npoints = 0
        if self.divided:
            npoints += len(self.nw) + len(self.ne) + len(self.se) + len(self.sw)
        return npoints
