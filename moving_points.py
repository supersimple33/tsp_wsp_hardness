import random

import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.interpolate as inter
import numpy as np

from wsp import util
from wsp import ds
from wsp import tsp

QTREE = ds.PKPRQuadTree

# MARK: VISUAL PARAMETERS
LW = 1.5 # Line width
KEEP_ASPECT = not True # Keep aspect ratio of plot
# MARK: SETUP
N_LEFT  = 4 # How many points in the left cluster
N_RIGHT = 4 # How many points in the right cluster
N_MIDDLE = 3
RADIUS_LEFT = 1.0
RADIUS_RIGHT = 1.0
OFFSET_RIGHT = 10.0
left_points : list[ds.Point] = util.generate_points(N_LEFT, lambda: (random.uniform(-RADIUS_LEFT, RADIUS_LEFT), random.uniform(-RADIUS_LEFT, RADIUS_LEFT))) # Generate points
right_points : list[ds.Point] = util.generate_points(N_RIGHT, lambda: (random.uniform(-RADIUS_RIGHT, RADIUS_RIGHT) + OFFSET_RIGHT, random.uniform(-RADIUS_RIGHT, RADIUS_RIGHT)))
middle_points = util.generate_points(3, lambda: (random.uniform(-RADIUS_RIGHT, RADIUS_RIGHT) + 2.5, random.uniform(-RADIUS_RIGHT, RADIUS_RIGHT) + 2.5))

# left_points = [ds.Point(0,1), ds.Point(0,-1), ds.Point(0.739,0)] # ~0.739 is the decision boundary at 30 points away???
# right_points = [ds.Point(9, 0.01)] # [ds.Point(3.1,0.01)]
# left_points = [ds.Point(0,1), ds.Point(0,-1), ds.Point(1,1), ds.Point(1,-1), ds.Point(0.25,0.25), ds.Point(1,0)]
# right_points = [ds.Point(10,0), ds.Point(10,0.5)]

fig, ax = plt.subplots(1, 1, figsize=(22,10)) # sharex = True?
ax.set_yscale('linear')

# MARK: Travelling Salesman Problems
ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, left_points + right_points + middle_points, np.array([None, ax]), s=2.0)
left_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, left_points, np.array([None, None]), s=2.0)
right_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, right_points, np.array([None, None]), s=2.0)
middle_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, middle_points, np.array([None, None]), s=2.0)

# MARK: INTERACTIONS
held_point : ds.Point  = None
pointA : ds.Point = None
pointB : ds.Point = None
def button_press_callback(event):
    global held_point, pointA, pointB
    if  event.inaxes is None:
        return
    pos = ds.Point(event.xdata, event.ydata)

    if event.button == 3:
        if pointA is None and pointB is None:
            pointA = util.closest_point(pos, ts_problem.points)
            plt.title(f"Point A: {pointA}")
        elif pointB is None:
            pointB = util.closest_point(pos, ts_problem.points)
            print("Point A:", pointA, "Point B:", pointB)
            plt.title(f"Distance: {pointA.distance_to(pointB)}")
            plt.plot([pointA.x, pointB.x], [pointA.y, pointB.y])
        else:
            pointA = None
            pointB = None
            plt.title("")
        plt.pause(0.1)
        return
    held_point = None

    closest = util.closest_point(pos, ts_problem.points)
    if closest.distance_to(pos) > ts_problem.quadtree.boundary.diameter()*0.05:
        return

    held_point = closest
    print("Grabbed:", held_point)
def button_release_callback(event):
    global held_point, ts_problem, left_problem, right_problem
    if event.button != 1 or event.inaxes is None or held_point is None:
        return
    new_point = ds.Point(event.xdata, event.ydata)

    ax.clear()
    if held_point in left_points:
        left_points.remove(held_point)
        left_points.append(new_point)
        left_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, left_points, np.array([None, None]), s=2.0)
    else:
        right_points.remove(held_point)
        right_points.append(new_point)
        right_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, right_points, np.array([None, None]), s=2.0)

    ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, left_points + right_points, np.array([None, ax]), s=2.0)
    ts_problem.draw_tour(ts_problem.dp_path[0], 'y', '-', linewidth=LW)
    ts_problem.draw_tour(left_problem.dp_path[0], 'r', (0, (5, 10)), linewidth=LW)
    ts_problem.draw_tour(right_problem.dp_path[0], 'm', (0, (5, 10)), linewidth=LW)
    plt.pause(0.1)

    # ts_problem.points[held_point] = pos
    # ts_problem.update()
    # ts_problem.draw_tour(ts_problem.dp_path[0])

    held_point = None

fig.canvas.mpl_connect('button_press_event', button_press_callback) # hook up interactions
fig.canvas.mpl_connect('button_release_event', button_release_callback) # hook up interactions

ts_problem.draw_tour(ts_problem.dp_path[0], 'y', '-', linewidth=LW)
ts_problem.draw_tour(left_problem.dp_path[0], 'r', (0, (5, 10)), linewidth=LW)
ts_problem.draw_tour(right_problem.dp_path[0], 'm', (0, (5, 10)),linewidth=LW)
ts_problem.draw_tour(middle_problem.dp_path[0], 'g', (0, (5, 10)), linewidth=LW)

if KEEP_ASPECT:
    plt.gca().set_aspect("equal")
plt.show()
