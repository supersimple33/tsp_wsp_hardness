import random
from random import randrange

import numpy as np
from deprecation import deprecated, fail_if_not_removed

from wsp import ds

def load_points(filename: str, shuffle=True) -> list[ds.Point]:
    """Read just the points from the file, don't do anything else. 
    Will shuffle points so the same tree isn't continually generated."""
    points = []

    with open(filename, 'r') as f: # reads .TSP files
        line = f.readline()
        mode = "start"
        while line != '':  # The EOF char is an empty string
            if line == "EOF\n":
                break
            if mode == "start": # Read in metadata from the file
                if line[len(line) - 1] == "\n":
                    line = line[:-1]
                if len(line) > 7 and line[:7] == "bounds:":
                    bounds = [int(i) for i in line[7:].split()]
                if line == "NODE_COORD_SECTION":
                    mode = "node"
                #if len(line) == 0 or line[0] == '#': # ignores empty lines and #comments
                line = f.readline()
                continue
            # start reading node coords
            if mode == "node":
                if line == "TOUR_SECTION\n":
                    mode = "tour"
                    break
                splitLine = line.split()
                if len(splitLine) == 3:
                    splitLine = splitLine[1:]
                p = ds.Point(float(splitLine[0].strip()), float(splitLine[1].strip()))
                points.append(p)
                line = f.readline()

    if shuffle:
        random.shuffle(points)

    return points

@deprecated("Use load_points instead")
def loadFromFile(filename: str, do_offset=False) -> list[ds.Point]:
    """Read the points from the file, and jiggle the points if do_offset"""
    print("Deprecated: Use load_points instead")
    points = []
    # read points from file
    bounds = []
    offset = [1020, 1890] # jittering dataset by epsilon 1 and 2
    with open(filename, 'r') as f: # reads .TSP files
        line = f.readline()
        mode = "start"
        while line != '':  # The EOF char is an empty string
            if line == "EOF\n":
                break
            if mode == "start":
                if line[len(line) - 1] == "\n":
                    line = line[:-1]
                if len(line) > 7 and line[:7] == "bounds:":
                    bounds = [int(i) for i in line[7:].split()]
                if line == "NODE_COORD_SECTION":
                    mode = "node"
                #if len(line) == 0 or line[0] == '#': # ignores empty lines and #comments
                line = f.readline()
                continue
            # start reading node coords
            if mode == "node":
                if line == "TOUR_SECTION\n":
                    mode = "tour"
                    break
                splitLine = line.split()
                if len(splitLine) == 3:
                    splitLine = splitLine[1:]
                p = ds.Point(float(splitLine[0].strip()), float(splitLine[1].strip()))
                points.append(p)
                line = f.readline()

    # find boundaries
    minX = float('inf')
    minY = float('inf')
    maxX = float('-inf')
    maxY = float('-inf')
    for p in points:
        if p.x < minX:
            minX = p.x
        if p.y < minY:
            minY = p.y
        if p.x > maxX:
            maxX = p.x
        if p.y > maxY:
            maxY = p.y
    minX -= 1.1 # NOTE: Why are we doing this?
    minY -= 1.1
    maxX += 1.1
    maxY += 1.1
    if do_offset: # NOTE: Why does offset only run on min/max?
        minX -= randrange(50)
        minY -= randrange(50)
        maxX += randrange(50)
        maxY += randrange(50)

    random.shuffle(points)

    return (points, minX, minY, maxX, maxY)
