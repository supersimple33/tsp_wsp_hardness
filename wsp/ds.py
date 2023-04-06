import numpy as np
from wsp import util

class Point:
    #A point located at (x,y) in 2D space.
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return '{}'.format(str((self.x, self.y)))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)
    def __add__(self, o):
        return Point(self.x + o.x, self.y + o.y)
    def __sub__(self, o):
        return Point(self.x - o.x, self.y - o.y)
    def __truediv__(self, o):
        return Point(self.x / o, self.y / o)
    def to_list(self):
        return [self.x, self.y]
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

def min_dist(block_A, block_B):
    min_p1, min_p2 = min_proj(block_A, block_B)
    return min_p1.distance_to(min_p2)

def min_proj(block_A, block_B):
  """Min dist between points from Quadtree block_A and Quadtree block_B"""
  set_A = block_A.get_points()
  set_B = block_B.get_points()
  #print(set_A, set_B)
  return util.min_proj(set_A, set_B)#min_p1, min_p2

def shrink_boundaries(block, regular=True):
  points = block.get_points()
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
  minX -= 0.1
  minY -= 0.1
  maxX += 0.1
  maxY += 0.1
  #print(minX, minY, maxX, maxY)
  block.boundary = Rect(minX,minY,maxX,maxY)
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

# PK PR QUADTREE

class PKPMRQuadTree:
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        self.boundary = boundary # boundary of current block
        self.bucket = bucket
        self.ax = ax
        self.depth = depth # mostly for string visualization spacing
        self.points = [] # center point
        self.children = [] # includes points and nodes
        self.connection = [] # WSP connection
        self.divided = False # flag for if divided into 4 child quads
        self.pk_aggregated = False # flag for if aggregated
        self.leaf = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        if self.pk_aggregated:
          sp = ' ' * self.depth * 2
          s = str(self.boundary) + ' --> ' + str(self.points) 
          #print(self.depth, len(self.children))
          for c in self.children:
            s += '\n' + sp + 'child:' + str(c)
          return s
        else:
          sp = ' ' * self.depth * 2
          s = str(self.boundary) + ' --> ' + str(self.points) 
          if not self.divided:
              return s
          return s + '\n' + '\n'.join([
                  sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                  sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def str_short(self):
      return str(self.get_points()) #str(self.boundary) + 

    def diameter(self):
      return self.boundary.diameter()
    def center(self):
      return self.boundary.center()

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        mid = Point((self.boundary.xMin + self.boundary.xMax) / 2, (self.boundary.yMin + self.boundary.yMax) / 2)
        self.nw = PKPMRQuadTree(Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax), self.ax, self.bucket, self.depth + 1)
        self.ne = PKPMRQuadTree(Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax), self.ax, self.bucket,  self.depth + 1)
        self.se = PKPMRQuadTree(Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y), self.ax, self.bucket, self.depth + 1)
        self.sw = PKPMRQuadTree(Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y), self.ax, self.bucket, self.depth + 1)
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
          #self.insert(p, True)
          self.ne.insert(p, True)
          self.nw.insert(p, True)
          self.se.insert(p, True)
          self.sw.insert(p, True)
        # draw
        self.ax[0].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[0].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")
        self.ax[1].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="lightgray")
        self.ax[1].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="lightgray")

    def insert(self, point, no_divide=False):
        """Try to insert Point point into this QuadTree."""
        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
          self.points.append(point)
          if not no_divide and len(self.points) > 1:
            self.divide()

          return True

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def get_points_rec(self, found_points):
        """Find the points in the quadtree that lie within boundary."""
        for point in self.points:
            found_points.append(point)

        # if this node has children, search them too.
        for child in self.children:
          child.get_points_rec(found_points)
          
        return found_points

    def get_points(self):
      return self.get_points_rec([])

    def pk_draw(self):
      for child in self.children:
        if len(child) > 1:
          child.pk_draw()
        #elif child.leaf:
        self.ax[1].plot([child.boundary.xMin, child.boundary.xMax],[child.boundary.yMin, child.boundary.yMin], color="blue")
        self.ax[1].plot([child.boundary.xMin, child.boundary.xMax],[child.boundary.yMax, child.boundary.yMax], color="blue")
        self.ax[1].plot([child.boundary.xMin, child.boundary.xMin],[child.boundary.yMin, child.boundary.yMax], color="blue")
        self.ax[1].plot([child.boundary.xMax, child.boundary.xMax],[child.boundary.yMin, child.boundary.yMax], color="blue")

    def pk_aggregate(self, k, parent=None):
      # removes k-empty nodes and reassigns to grandparents
      self.pk_aggregated = True

      if self.divided:
        rec_children = []
        rec_children.append(self.nw.pk_aggregate(k, self))
        rec_children.append(self.ne.pk_aggregate(k, self))
        rec_children.append(self.se.pk_aggregate(k, self))
        rec_children.append(self.sw.pk_aggregate(k, self))
        for c in rec_children:
          if c != None:
            self.children.append(c)

        #print(len(self.children), rec_children)
        if len(self.children) == 0:
          self.divided = False

        self.nw = None
        self.ne = None
        self.se = None
        self.sw = None

        if parent != None:
          if len(self) < k:
            # pass children upwards
            #print("len", len(self), k)
            parent.children += self.children
            return None
          else:
            return self
      else:
        #print("leaf node", len(self.points))
        self.leaf = True
        if len(self.points) > 0:
          return self
        else:
          return None
      
      return self

    def __len__(self):
        """Return the number of points in the quadtree."""
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw) if self.nw != None else 0
            npoints += len(self.ne) if self.ne != None else 0
            npoints += len(self.se) if self.se != None else 0
            npoints += len(self.sw) if self.sw != None else 0
        for c in self.children:
          npoints += len(c)
        return npoints

# PK PR QUADTREE

class PKPRQuadTree:
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        self.boundary = boundary # boundary of current block
        self.bucket = bucket
        self.ax = ax
        self.depth = depth # mostly for string visualization spacing
        self.points = [] # center point
        self.children = [] # includes points and nodes
        self.connection = [] # WSP connection
        self.divided = False # flag for if divided into 4 child quads
        self.pk_aggregated = False # flag for if aggregated
        self.leaf = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        if self.pk_aggregated:
          sp = ' ' * self.depth * 2
          s = str(self.boundary) + ' --> ' + str(self.points) 
          #print(self.depth, len(self.children))
          for c in self.children:
            s += '\n' + sp + 'child:' + str(c)
          return s
        else:
          sp = ' ' * self.depth * 2
          s = str(self.boundary) + ' --> ' + str(self.points) 
          if not self.divided:
              return s
          return s + '\n' + '\n'.join([
                  sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                  sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def str_short(self):
      return str(self.get_points()) #str(self.boundary) + 

    def diameter(self):
      return self.boundary.diameter()
    def center(self):
      return self.boundary.center()

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        mid = Point((self.boundary.xMin + self.boundary.xMax) / 2, (self.boundary.yMin + self.boundary.yMax) / 2)
        self.nw = PKPRQuadTree(Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax), self.ax, self.bucket, self.depth + 1)
        self.ne = PKPRQuadTree(Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax), self.ax, self.bucket,  self.depth + 1)
        self.se = PKPRQuadTree(Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y), self.ax, self.bucket, self.depth + 1)
        self.sw = PKPRQuadTree(Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y), self.ax, self.bucket, self.depth + 1)
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
          self.insert(p)
        # draw
        self.ax[0].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[0].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")
        self.ax[1].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="lightgray")
        self.ax[1].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="lightgray")

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
          if len(self.points) < self.bucket:
              # Node doesn't have a point yet.
              self.points.append(point)
              return True

          # Already leaf: divide if necessary, then try the sub-quads.
          self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def get_points_rec(self, found_points):
        """Find the points in the quadtree that lie within boundary."""
        for point in self.points:
            found_points.append(point)

        # if this node has children, search them too.
        for child in self.children:
          child.get_points_rec(found_points)
          
        return found_points

    def get_points(self):
      return self.get_points_rec([])

    def pk_draw(self):
      for child in self.children:
        if len(child) > 1:
          child.pk_draw()
        #elif child.leaf:
        self.ax[1].plot([child.boundary.xMin, child.boundary.xMax],[child.boundary.yMin, child.boundary.yMin], color="blue")
        self.ax[1].plot([child.boundary.xMin, child.boundary.xMax],[child.boundary.yMax, child.boundary.yMax], color="blue")
        self.ax[1].plot([child.boundary.xMin, child.boundary.xMin],[child.boundary.yMin, child.boundary.yMax], color="blue")
        self.ax[1].plot([child.boundary.xMax, child.boundary.xMax],[child.boundary.yMin, child.boundary.yMax], color="blue")

    def pk_aggregate(self, k, parent=None):
      # removes k-empty nodes and reassigns to grandparents
      self.pk_aggregated = True

      if self.divided:
        rec_children = []
        rec_children.append(self.nw.pk_aggregate(k, self))
        rec_children.append(self.ne.pk_aggregate(k, self))
        rec_children.append(self.se.pk_aggregate(k, self))
        rec_children.append(self.sw.pk_aggregate(k, self))
        for c in rec_children:
          if c != None:
            self.children.append(c)

        #print(len(self.children), rec_children)
        if len(self.children) == 0:
          self.divided = False

        self.nw = None
        self.ne = None
        self.se = None
        self.sw = None

        if parent != None:
          if len(self) < k:
            # pass children upwards
            #print("len", len(self), k)
            parent.children += self.children
            return None
          else:
            return self
      else:
        #print("leaf node", len(self.points))
        self.leaf = True
        if len(self.points) > 0:
          return self
        else:
          return None
      
      return self

    def __len__(self):
        """Return the number of points in the quadtree."""
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw) if self.nw != None else 0
            npoints += len(self.ne) if self.ne != None else 0
            npoints += len(self.se) if self.se != None else 0
            npoints += len(self.sw) if self.sw != None else 0
        for c in self.children:
          npoints += len(c)
        return npoints


# PMR QUADTREE

class PMRQuadTree:
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=-1, depth=0):
        """Initialize this node of the quadtree."""
        self.boundary = boundary # boundary of current block
        self.bucket = bucket
        self.ax = ax
        self.depth = depth # mostly for string visualization spacing
        self.points = [] # center point
        self.connection = [] # WSP connection
        self.divided = False # flag for if divided into 4 child quads

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + ' --> ' + str(self.points) 
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
        mid = Point((self.boundary.xMin + self.boundary.xMax) / 2, (self.boundary.yMin + self.boundary.yMax) / 2)
        self.nw = PMRQuadTree(Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax), self.ax, self.bucket, self.depth + 1)
        self.ne = PMRQuadTree(Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax), self.ax, self.bucket,  self.depth + 1)
        self.se = PMRQuadTree(Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y), self.ax, self.bucket, self.depth + 1)
        self.sw = PMRQuadTree(Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y), self.ax, self.bucket, self.depth + 1)
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
          #self.insert(p, True)
          self.ne.insert(p, True)
          self.nw.insert(p, True)
          self.se.insert(p, True)
          self.sw.insert(p, True)
        # draw
        self.ax[0].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[0].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")
        self.ax[1].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[1].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")

    def insert(self, point, no_divide=False):
        """Try to insert Point point into this QuadTree."""
        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
          self.points.append(point)
          if not no_divide and len(self.points) > 1:
            self.divide()

          return True

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def get_points_rec(self, found_points):
        """Find the points in the quadtree that lie within boundary."""
        for point in self.points:
            found_points.append(point)

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
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints


# POINT REGION QUADTREE

class PRQuadTree:
    """Point Region Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=1, depth=0):
        """Initialize this node of the quadtree."""
        self.boundary = boundary # boundary of current block
        self.bucket = bucket
        self.ax = ax
        self.depth = depth # mostly for string visualization spacing
        self.points = [] # center point
        self.connection = [] # WSP connection
        self.divided = False # flag for if divided into 4 child quads

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + ' --> ' + str(self.points) 
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def str_short(self):
      return str(self.get_points()) #str(self.boundary) + 

    def diameter(self):
      return self.boundary.diameter()
    def center(self):
      return self.boundary.center()

    def divide(self):
        """Divide (branch) this node by spawning four children nodes around a point."""
        mid = Point((self.boundary.xMin + self.boundary.xMax) / 2, (self.boundary.yMin + self.boundary.yMax) / 2)
        self.nw = PRQuadTree(Rect(self.boundary.xMin, mid.y, mid.x, self.boundary.yMax), self.ax, self.bucket, self.depth + 1)
        self.ne = PRQuadTree(Rect(mid.x, mid.y, self.boundary.xMax, self.boundary.yMax), self.ax, self.bucket,  self.depth + 1)
        self.se = PRQuadTree(Rect(mid.x, self.boundary.yMin, self.boundary.xMax, mid.y), self.ax, self.bucket, self.depth + 1)
        self.sw = PRQuadTree(Rect(self.boundary.xMin, self.boundary.yMin, mid.x, mid.y), self.ax, self.bucket, self.depth + 1)
        self.divided = True
        # reinsert point
        points_to_reinsert = self.points
        self.points = []
        for p in points_to_reinsert:
          self.insert(p)
        # draw
        self.ax[0].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[0].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")
        self.ax[1].plot([mid.x, mid.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[1].plot([self.boundary.xMin, self.boundary.xMax],[mid.y, mid.y], color="gray")

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False

        if not self.divided:
          if len(self.points) < self.bucket:
              # Node doesn't have a point yet.
              self.points.append(point)
              return True

          # Already leaf: divide if necessary, then try the sub-quads.
          self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def get_points_rec(self, found_points):
        """Find the points in the quadtree that lie within boundary."""
        for point in self.points:
            found_points.append(point)

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
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

# POINT QUADTREE ##############################################################
class PointQuadTree:
    """Point Quadtree implementation."""

    def __init__(self, boundary, ax, bucket=-1, depth=0):
        """Initialize this node of the quadtree."""
        self.boundary = boundary # boundary of current block
        self.ax = ax
        self.depth = depth # mostly for string visualization spacing
        self.point = None # center point
        self.connection = [] # WSP connection
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
        self.nw = PointQuadTree(Rect(self.boundary.xMin, self.point.y, self.point.x, self.boundary.yMax), self.ax, self.depth + 1)
        self.ne = PointQuadTree(Rect(self.point.x, self.point.y, self.boundary.xMax, self.boundary.yMax), self.ax, self.depth + 1)
        self.se = PointQuadTree(Rect(self.point.x, self.boundary.yMin, self.boundary.xMax, self.point.y), self.ax, self.depth + 1)
        self.sw = PointQuadTree(Rect(self.boundary.xMin, self.boundary.yMin, self.point.x, self.point.y), self.ax, self.depth + 1)
        self.divided = True
        # draw
        self.ax[0].plot([self.point.x, self.point.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[0].plot([self.boundary.xMin, self.boundary.xMax],[self.point.y, self.point.y], color="gray")
        self.ax[1].plot([self.point.x, self.point.x],[self.boundary.yMin, self.boundary.yMax], color="gray")
        self.ax[1].plot([self.boundary.xMin, self.boundary.xMax],[self.point.y, self.point.y], color="gray")

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