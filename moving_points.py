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

# MARK: SETUP
N = 3 # How many points in the main cluster
RADIUS = 2.0
points : list[ds.Point] = util.generate_points(N, lambda: (random.uniform(-RADIUS, RADIUS), random.uniform(-RADIUS, RADIUS))) + [ds.Point(30, 0)] # Generate points

fig, ax = plt.subplots(1, 1, figsize=(18,9)) # sharex = True?
ax.set_yscale('linear')

ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, points, np.array([None, ax]), s=2.0)

# MARK: INTERACTIONS
held_point = None
pointA = None
pointB = None
def button_press_callback(event):
    global held_point, pointA, pointB
    if  event.inaxes is None:
        return
    pos = ds.Point(event.xdata, event.ydata)

    if event.button == 3:
        if pointA is None and pointB is None:
            pointA = util.closest_point(pos, ts_problem.points)
            plt.title("Point A: " + str(pointA))
            plt.pause(0.1)
        elif pointB is None:
            pointB = util.closest_point(pos, ts_problem.points)
            print("Point A:", pointA, "Point B:", pointB)
            plt.title("Distance: " + str(pointA.distance_to(pointB)))
            plt.pause(0.1)
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

    held_point = ts_problem.points.index(closest)
    print("Grabbed:", held_point)
def button_release_callback(event):
    global held_point
    global ts_problem
    if event.button != 1 or event.inaxes is None or held_point is None:
        return
    pos = ds.Point(event.xdata, event.ydata)

    new_points = ts_problem.points.copy()
    new_points[held_point] = pos
    ax.clear()
    ts_problem = tsp.TravellingSalesmanProblem[QTREE](QTREE, new_points, np.array([None, ax]), s=2.0)
    ts_problem.draw_tour(ts_problem.dp_path[0])
    plt.pause(0.1)

    # ts_problem.points[held_point] = pos
    # ts_problem.update()
    # ts_problem.draw_tour(ts_problem.dp_path[0])

    held_point = None

fig.canvas.mpl_connect('button_press_event', button_press_callback) # hook up interactions
fig.canvas.mpl_connect('button_release_event', button_release_callback) # hook up interactions

ts_problem.draw_tour(ts_problem.dp_path[0])

plt.show()
