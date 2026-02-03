import math
import random
from typing import Optional
from itertools import permutations
from collections import defaultdict

from deprecation import deprecated

from wsp import ds

# MARK: Distances


def euclid_dist(p1: "ds.Point", p2: "ds.Point") -> float:
    """Euclidean distance between two points"""
    return math.sqrt(((p2.x - p1.x) ** 2) + ((p2.y - p1.y) ** 2))


# MARK: Projection


def sublist_get_points(lst) -> list["ds.Point"]:
    """Flatten list of arbitrary lists of points to list of points"""
    points = []
    for item in lst:
        if isinstance(item, list):
            points += sublist_get_points(item)
        else:
            points.append(item)
    return points


@deprecated(
    "This function should be restructured to cleanup its meaning and return type"
)
def min_proj_set_or_point(item_A, item_B, min2=False):
    if not isinstance(item_A, list):
        item_A = [item_A]
    if not isinstance(item_B, list):
        item_B = [item_B]
    item_A = sublist_get_points(item_A)
    item_B = sublist_get_points(item_B)

    if min2:  # TODO: sourcery refactor
        return min2_proj(item_A, item_B)
    return min_proj(item_A, item_B)


def min_proj(
    set_A: list["ds.Point"], set_B: list["ds.Point"]
) -> tuple["ds.Point", "ds.Point"]:
    """Min pair between points from set_A and set_B"""
    """avg_A = ds.Point(0,0)
    avg_B = ds.Point(0,0)
    for p_A in set_A:
        avg_A += p_A
    for p_B in set_B:
        avg_B += p_B
    avg_A /= len(set_A)
    avg_B /= len(set_B)

    min_d1 = 99999999
    min_d2 = 99999999
    for p_A in set_A:
        dist = avg_B.distance_to(p_A)
        if dist < min_d1:
            min_p1 = p_A
            min_d1 = dist
    for p_B in set_B:
        dist = avg_A.distance_to(p_B)
        if dist < min_d2:
            min_p2 = p_B
            min_d2 = dist"""

    if len(set_A) == 0 or len(set_B) == 0:
        raise ValueError("Empty Quadtree block")

    mind = 9999999999
    min_p1 = None
    min_p2 = None
    for p_A in set_A:
        for p_B in set_B:
            dist = p_A.distance_to(p_B)
            if dist < mind:
                mind = dist
                min_p1 = p_A
                min_p2 = p_B

    return min_p1, min_p2


def min2_proj(
    set_A, set_B
) -> tuple["ds.Point", "ds.Point", Optional["ds.Point"], Optional["ds.Point"]]:
    """Min pair between points from set_A and set_B"""
    avg_A = ds.Point(0, 0)
    avg_B = ds.Point(0, 0)
    for p_A in set_A:
        avg_A += p_A
    for p_B in set_B:
        avg_B += p_B
    avg_A /= len(set_A)
    avg_B /= len(set_B)

    """min_d1 = 99999999
    min_d2 = 99999999
    min_p1 = None
    min_p2 = None
    for p_A in set_A:
        #print("fing", p_A)
        dist = avg_B.distance_to(p_A)
        if dist < min_d1:
            sec_p1 = min_p1
            min_p1 = p_A
            min_d1 = dist
        elif sec_p1 == None:
            sec_p1 = p_A
    for p_B in set_B:
        dist = avg_A.distance_to(p_B)
        if dist < min_d2:
            sec_p2 = min_p2
            min_p2 = p_B
            min_d2 = dist
        elif sec_p2 == None:
            sec_p2 = p_B"""

    mind = 99999999
    min_p1 = None
    min_p2 = None
    sec_p1 = None
    sec_p2 = None
    rank = []
    # print(set_A, set_B)
    for p_A in set_A:
        for p_B in set_B:
            dist = p_A.distance_to(p_B)
            inserted = False
            if len(rank) > 0:
                for i, tup in enumerate(rank):
                    if dist < tup[2]:
                        rank.insert(i, (p_A, p_B, dist))
                        inserted = True
                        break
            if not inserted:
                rank.append((p_A, p_B, dist))

            """if dist < mind:
                mind = dist
                if min_p1 != None and avg_B.distance_to(p_A) < avg_B.distance_to(min_p1): #p_A != min_p1:
                    sec_p1 = min_p1
                if min_p2 != None and avg_A.distance_to(p_B) < avg_A.distance_to(min_p2): #p_B != min_p2:
                    sec_p2 = min_p2
                min_p1 = p_A
                min_p2 = p_B"""
    min_p1 = rank[0][0]
    min_p2 = rank[0][1]
    i = 1
    while i < len(rank) and (sec_p1 is None or sec_p2 is None):
        # print("searching for", i, rank[i])
        if sec_p1 is None and rank[i][0] != min_p1:
            sec_p1 = rank[i][0]
        if sec_p2 is None and rank[i][1] != min_p2:
            sec_p2 = rank[i][1]
        i += 1
    # print("min proj 2", min_p1, min_p2, sec_p1, sec_p2)
    return min_p1, min_p2, sec_p1, sec_p2


def closest_point(point: "ds.Point", points: list["ds.Point"]) -> "ds.Point":
    # REVIEW: this can be speedup using Samet's algorithm
    """Closest point to point in points"""
    min_dist = float("inf")
    min_point = None
    for p in points:
        dist = point.distance_to(p)
        if dist < min_dist:
            min_dist = dist
            min_point = p
    return min_point


# MARK: Point Generation


def generate_points(
    n: int, generator=lambda: (random.uniform(-100, 100), random.uniform(-100, 100))
) -> list["ds.Point"]:
    """Generate n random points"""
    points = []
    for i in range(n):
        points.append(ds.Point(*generator()))
    return points


def circle_points(
    n: int, radius: float, center: Optional["ds.Point"], rad_offset=0.79
) -> list["ds.Point"]:
    """Generate n points on a circle"""

    if center is None:  # avoids circular import
        center = ds.Point(0, 0)

    points = []
    for i in range(n):
        points.append(
            ds.Point(
                center.x + radius * math.cos(rad_offset + (2 * math.pi * i / n)),
                center.y + radius * math.sin(rad_offset + (2 * math.pi * i / n)),
            )
        )
    return points


# MARK: Path identification


# TODO: Convert to DP implementation
def hamiltonian_path(
    start: "ds.Point", end: "ds.Point", points: list["ds.Point"]
) -> list["ds.Point"]:
    """Finds the shortest path between start and end which visits every point exactly once"""
    assert start in points and end in points and start != end and len(points) >= 2

    if len(points) == 2:
        return (start, end)

    min_path = []
    min_dist = float("inf")

    for perm in permutations(filter(lambda p: p not in (start, end), points)):
        dist = (
            euclid_dist(start, perm[0]) + calc_dist(perm) + euclid_dist(perm[-1], end)
        )
        if dist < min_dist:
            min_dist = dist
            min_path = (start,) + perm + (end,)

    assert (
        min_path is not None
        and len(min_path) == len(points)
        and min_path[0] == start
        and min_path[-1] == end
    )
    return min_path


def group_by(l, key=lambda x: x, value=lambda x: x):
    d = defaultdict(list)
    for item in l:
        d[key(item)].append(value(item))
    return d
