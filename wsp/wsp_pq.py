import numpy as np
import sys
import matplotlib.pyplot as plt

class Point:
    #A point located at (x,y) in 2D space.
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return '{}'.format(str((self.x, self.y)))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)
    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
    def __init__(self, xMin, yMin, xMax, yMax):
      self.xMin = xMin
      self.yMin = yMin
      self.xMax = xMax
      self.yMax = yMax

    def __repr__(self):
        return str((self.xMin,self.yMin, self.xMax, self.yMax))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.xMin,
                    self.yMin, self.xMax, self.yMax)

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""
        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.xMin and
                point_x <  self.xMax and
                point_y >= self.yMin and
                point_y < self.yMax)

    def diameter(self):
      # diagonal
      return np.hypot(self.xMax - self.xMin, self.yMax - self.yMin)
    def center(self):
      return ((self.xMax + self.xMin) / 2, (self.yMax + self.yMin) / 2)

class QuadTree:
    """Point Quadtree implementation."""

    def __init__(self, boundary, depth=0):
        """Initialize this node of the quadtree."""
        self.boundary = boundary # boundary of current block
        self.depth = depth # mostly for string visualization spacing
        self.point = None # center point
        self.connection = None # WSP connection
        self.divided = False # flag for if divided into 4 child quads

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + ' --> ' + str(self.point) 
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def str_short(self):
      return str(self.boundary)

    def diameter(self):
      return self.boundary.diameter()
    def center(self):
      return self.boundary.center()

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        self.nw = QuadTree(Rect(self.boundary.xMin, self.point.y, self.point.x, self.boundary.yMax), self.depth + 1)
        self.ne = QuadTree(Rect(self.point.x, self.point.y, self.boundary.xMax, self.boundary.yMax), self.depth + 1)
        self.se = QuadTree(Rect(self.point.x, self.boundary.yMin, self.boundary.xMax, self.point.y), self.depth + 1)
        self.sw = QuadTree(Rect(self.boundary.xMin, self.boundary.yMin, self.point.x, self.point.y), self.depth + 1)
        self.divided = True

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False

        if self.point == None:
            # Node doesn't have a point yet.
            self.point = point
            return True

        # Already leaf: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def get_points_rec(self, found_points):
        """Find the points in the quadtree that lie within boundary."""
        if self.point != None:
            found_points.append(self.point)

        # if this node has children, search them too.
        if self.divided:
          self.nw.get_points_rec(found_points)
          self.ne.get_points_rec(found_points)
          self.se.get_points_rec(found_points)
          self.sw.get_points_rec(found_points)
        return found_points

    def get_points(self):
      return self.get_points_rec([])

    def __len__(self):
        """Return the number of points in the quadtree."""
        if self.point != None:
          npoints = 1
        else:
          npoints = 0
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

def mindist(block_A, block_B):
  """Min dist between points from Quadtree block_A and Quadtree block_B"""
  points_A = block_A.get_points()
  points_B = block_B.get_points()
  mind = 99999999
  for p_A in points_A:
    for p_B in points_B:
      dist = p_A.distance_to(p_B)
      if dist < mind:
        mind = dist
  return mind

def plotPoints(points):
  x = []
  y = []
  for p in points:
    x.append(p.x)
    y.append(p.y)

  fig, ax = plt.subplots()
  #fig = plt.scatter(x, y)
  return fig, ax
  #plt.draw()
  #plt.show(block=False)

def runWSP(filename, s, debug):
  points = []
  # read points from file
  #bounds = []
  with open(filename, 'r') as f:
     line = f.readline()
     while line != '':  # The EOF char is an empty string
        if line[len(line) - 1] == "\n":
          line = line[:-1]
        if len(line) == 0 or line[0] == '#': # ignores empty lines and #comments
          line = f.readline()
          continue
        if len(line) > 7 and line[:7] == "bounds:":
          #bounds = [int(i) for i in line[7:].split(",")]
          line = f.readline()
          continue
        splitLine = line.split(",")
        p = Point(float(splitLine[0].strip()), float(splitLine[1].strip()))
        points.append(p)
        line = f.readline()
  # find boundaries
  minX = sys.float_info.max
  minY = sys.float_info.max
  maxX = sys.float_info.min
  maxY = sys.float_info.min
  for p in points:
    if p.x < minX:
      minX = p.x
    if p.y < minY:
      minY = p.y
    if p.x > maxX:
      maxX = p.x
    if p.y > maxY:
      maxY = p.y

  # plot points
  fig = plt.figure(figsize=(6,6))
  
  ax = fig.add_subplot(1, 1, 1)
  x = []
  y = []
  for p in points:
    x.append(p.x)
    y.append(p.y)
  fig = plt.scatter(x, y)

  # build point quadtree, insert in order
  rootNode = QuadTree(Rect(minX,minY,maxX,maxY))
  for point in points:
    rootNode.insert(point)

  if (debug):
    print(points, "\n")
    print(rootNode, "\n\n")

  # WSP queue search
  #s = 1 # wsp separation factor
  plt.plot([rootNode.boundary.xMin, rootNode.boundary.xMax],[rootNode.boundary.yMin, rootNode.boundary.yMin], color="gray")
  plt.plot([rootNode.boundary.xMin, rootNode.boundary.xMax],[rootNode.boundary.yMax, rootNode.boundary.yMax], color="gray")
  plt.plot([rootNode.boundary.xMin, rootNode.boundary.xMin],[rootNode.boundary.yMin, rootNode.boundary.yMax], color="gray")
  plt.plot([rootNode.boundary.xMax, rootNode.boundary.xMax],[rootNode.boundary.yMin, rootNode.boundary.yMax], color="gray")
  queue = [(rootNode, rootNode)]
  while len(queue) > 0:
    pair = queue[0]
    queue = queue[1:]

    block_A, block_B = pair[0], pair[1]

    plt.plot([block_A.point.x, block_A.point.x],[block_A.boundary.yMin, block_A.boundary.yMax], color="gray")
    plt.plot([block_A.boundary.xMin, block_A.boundary.xMax],[block_A.point.y, block_A.point.y], color="gray")

    if len(block_A) == 0 or len(block_B) == 0:
      continue
    
    if mindist(block_A, block_B) >= s * block_A.diameter():
      if (debug):
        print("found a WSP: ", block_A.str_short(), " <~~~~~> ", block_B.str_short())
      block_A.connection = block_B
      block_B.connection = block_A
      circle1 = plt.Circle(block_A.center(), block_A.diameter() / 2, color='r', fill=False)
      circle2 = plt.Circle(block_B.center(), block_B.diameter() / 2, color='r', fill=False)
      ax.add_patch(circle1)
      ax.add_patch(circle2)
      #line
      plt.plot([block_A.center()[0], block_B.center()[0]],[block_A.center()[1], block_B.center()[1]])
      continue

    if block_A.diameter() > block_B.diameter():
      if (block_A.divided):
        queue.append((block_B, block_A.nw))
        queue.append((block_B, block_A.ne))
        queue.append((block_B, block_A.se))
        queue.append((block_B, block_A.sw))
    else:
      if (block_B.divided):
        queue.append((block_A, block_B.nw))
        queue.append((block_A, block_B.ne))
        queue.append((block_A, block_B.se))
        queue.append((block_A, block_B.sw))
  
  return rootNode