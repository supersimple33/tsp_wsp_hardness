import math
import statistics
from typing import Type

import matplotlib.pyplot as plt # type: ignore

from wsp import file_load
from wsp import ds

BUFFER = 1.1

# USES PR QUADTREE!
fig, ax = plt.subplots(1, 2, figsize=(12,6))


def runWSP(filename: str, s = 1.0, debug = False, shrink = False, quadtree : Type[ds.AbstractQuadTree] = ds.PMRQuadTree, bucket = 1) -> tuple[ds.AbstractQuadTree, int]:
    """Runs the WSP algorithm on the given file, with the given separation factor."""
    points = file_load.load_points(filename, True)
    return from_points(points, s, debug, shrink, quadtree, bucket)

# def from_points(points, s, debug, shrink, quadtree, bucket):
def from_points(points: list[ds.Point], s = 1.0, debug = False, shrink = False, quadtree : Type[ds.AbstractQuadTree] = ds.PMRQuadTree, bucket = 1) -> tuple[ds.AbstractQuadTree, int]:
    """Runs the WSP algorithm on the given list of points, with the given separation factor.

        Parameters:
            points (list[ds.Point]): The list of points to run the algorithm on.
            s (int): The separation factor.
            debug (bool): Whether to print debug info.
            shrink (bool): Whether to shrink the quadtree boundaries.
            quadtree (Type[ds.AbstractQuadTree]): The type of quadtree to use.
            bucket (int): The bucket size to use.

        Returns:
            tuple[ds.AbstractQuadTree, int]: The WSP tree and the number of WSPs found.
    """
    minX = min(p.x for p in points) - BUFFER
    minY = min(p.y for p in points) - BUFFER
    maxX = max(p.x for p in points) + BUFFER
    maxY = max(p.y for p in points) + BUFFER

    # build point quadtree, insert in order
    rootNode = quadtree(ds.Rect(minX, minY, maxX, maxY), ax, bucket)

    metric = 0.0
    total = 0
    sumsq = 0
    vals = []

    #rootNode = QuadTree(Rect(bounds[0],bounds[1],bounds[2],bounds[3]))
    for point in points:
        rootNode.insert(point)

    if debug:
        print(points, "\n")
        print(rootNode, "\n\n")


    wsp_count = 0

    # WSP queue search
    #s = 1 # wsp separation factor
    ax[0].plot([rootNode.boundary.xMin, rootNode.boundary.xMax],[rootNode.boundary.yMin, rootNode.boundary.yMin], color="gray")
    ax[0].plot([rootNode.boundary.xMin, rootNode.boundary.xMax],[rootNode.boundary.yMax, rootNode.boundary.yMax], color="gray")
    ax[0].plot([rootNode.boundary.xMin, rootNode.boundary.xMin],[rootNode.boundary.yMin, rootNode.boundary.yMax], color="gray")
    ax[0].plot([rootNode.boundary.xMax, rootNode.boundary.xMax],[rootNode.boundary.yMin, rootNode.boundary.yMax], color="gray")
    ax[1].plot([rootNode.boundary.xMin, rootNode.boundary.xMax],[rootNode.boundary.yMin, rootNode.boundary.yMin], color="gray")
    ax[1].plot([rootNode.boundary.xMin, rootNode.boundary.xMax],[rootNode.boundary.yMax, rootNode.boundary.yMax], color="gray")
    ax[1].plot([rootNode.boundary.xMin, rootNode.boundary.xMin],[rootNode.boundary.yMin, rootNode.boundary.yMax], color="gray")
    ax[1].plot([rootNode.boundary.xMax, rootNode.boundary.xMax],[rootNode.boundary.yMin, rootNode.boundary.yMax], color="gray")

    if quadtree == ds.PKPRQuadTree or quadtree == ds.PKPMRQuadTree:
        #print("PKPR aggregating")
        rootNode.pk_aggregate(2)
        if shrink:
            rootNode = ds.shrink_boundaries(rootNode, False)
        rootNode.pk_draw()
        #print(rootNode)
    else:
        if shrink:
            rootNode = ds.shrink_boundaries(rootNode)

    queue : list[tuple[ds.AbstractQuadTree, ds.AbstractQuadTree]] = [(rootNode, rootNode)] # (| None) needed for second element?
    while len(queue) > 0:
        pair = queue[0]
        queue = queue[1:]
        block_A, block_B = pair[0], pair[1]

        if block_A is None or block_B is None or len(block_A) == 0 or len(block_B) == 0: # NOTE: repeated calls to len are bad since i don't think it is cached
            continue

        points_A = block_A.covered_points
        points_B = block_B.covered_points
        if debug:
            print("considering WSP: ", block_A.str_short(), " <~~~~~> ", block_B.str_short(), (len(points_A) == 1 and len(points_B) == 1))
        if ds.min_dist(block_A, block_B) >= s * block_A.diameter or (len(points_A) == 1 and len(points_B) == 1 and not block_A.divided  and not block_B.divided and points_A[0] != points_B[0]):
            if debug:
                print("found a WSP: ", block_A.str_short(), " <~~~~~> ", block_B.str_short())
            wsp_count += 1
            block_A.connection.append(block_B)
            block_B.connection.append(block_A)
            circle1 = plt.Circle(block_A.center.to_tuple(), block_A.diameter / 2, color='r', fill=False)
            circle2 = plt.Circle(block_B.center.to_tuple(), block_B.diameter / 2, color='r', fill=False)
            ax[0].add_patch(circle1)
            ax[0].add_patch(circle2)
            #line
            ax[0].plot([block_A.center.x, block_B.center.x],[block_A.center.y, block_B.center.y])

            metric = (math.sqrt(((block_A.center.x - block_B.center.x) ** 2) + \
             ((block_A.center.y - block_B.center.y) ** 2))) #

            vals.append(metric)
            total += metric
            sumsq += metric*metric

            continue

        if issubclass(quadtree, ds.AbstractPKQuadTree): # PKPR or PKPMR
            if block_A.diameter > block_B.diameter:
                for child in block_A.children:
                    queue.append((block_B, child))
            else:
                for child in block_B.children:
                    queue.append((block_A, child))
        else:
            if block_A.diameter > block_B.diameter:
                if block_A.divided:
                    queue.append((block_B, block_A.nw))
                    queue.append((block_B, block_A.ne))
                    queue.append((block_B, block_A.se))
                    queue.append((block_B, block_A.sw))
            else:
                if block_B.divided:
                    queue.append((block_A, block_B.nw))
                    queue.append((block_A, block_B.ne))
                    queue.append((block_A, block_B.se))
                    queue.append((block_A, block_B.sw))

    # plot points
    x = []
    y = []
    for p in points:
        x.append(p.x)
        y.append(p.y)
    #fig = plt.scatter(x, y)
    ax[0].scatter(x, y)
    ax[1].scatter(x, y)

    vals = [2, 1] # FIXME: remove this
    avg_metric = sum(vals)/len(vals)
    var_metric = sumsq/len(vals) - (avg_metric*avg_metric)

    kurtosis = sum([(x-avg_metric)**4 for x in vals])/(len(vals)*var_metric**2)

    svals = [(x-avg_metric)/var_metric for x in vals]

    mean = sum(svals)/len(svals)

    tailed = (statistics.median(vals) - avg_metric)/(max(vals)-min(vals))

    return rootNode, wsp_count

# def plot_wsp
