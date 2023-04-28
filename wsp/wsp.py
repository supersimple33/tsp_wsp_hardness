from wsp import file_load
from wsp import ds
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import statistics

# USES PR QUADTREE!
fig, ax = plt.subplots(1, 2, figsize=(12,6))

def runWSP(filename, s, debug, shrink, quadtree, bucket):
  points, minX, minY, maxX, maxY = file_load.loadFromFile(filename, False)
  # build point quadtree, insert in order
  rootNode = quadtree(ds.Rect(minX,minY,maxX,maxY), ax, bucket)

  xRange = maxX - minX
  yRange = maxY - minY

  metric = 0
  total = 0
  sumsq = 0
  vals = []

  #rootNode = QuadTree(Rect(bounds[0],bounds[1],bounds[2],bounds[3]))
  for point in points:
    rootNode.insert(point)

  if (debug):
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

  queue = [(rootNode, rootNode)]
  while len(queue) > 0:
    pair = queue[0]
    queue = queue[1:]
    block_A, block_B = pair[0], pair[1]

    if block_A == None or block_B == None or len(block_A) == 0 or len(block_B) == 0:
      continue
    
    points_A = block_A.get_points()
    points_B = block_B.get_points()
    if (debug):
        print("considering WSP: ", block_A.str_short(), " <~~~~~> ", block_B.str_short(), (len(points_A) == 1 and len(points_B) == 1))
    if ds.min_dist(block_A, block_B) >= s * block_A.diameter() or (len(points_A) == 1 and len(points_B) == 1 and not block_A.divided  and not block_B.divided and points_A[0] != points_B[0]):
      if (debug):
        print("found a WSP: ", block_A.str_short(), " <~~~~~> ", block_B.str_short())
      wsp_count += 1
      block_A.connection.append(block_B)
      block_B.connection.append(block_A)
      circle1 = plt.Circle(block_A.center(), block_A.diameter() / 2, color='r', fill=False)
      circle2 = plt.Circle(block_B.center(), block_B.diameter() / 2, color='r', fill=False)
      ax[0].add_patch(circle1)
      ax[0].add_patch(circle2)
      #line
      ax[0].plot([block_A.center()[0], block_B.center()[0]],[block_A.center()[1], block_B.center()[1]])

      metric = (math.sqrt(((block_A.center()[0] - block_B.center()[0]) ** 2) + \
       ((block_A.center()[1] - block_B.center()[1]) ** 2))) #  

      vals.append(metric)
      total += metric
      sumsq += metric*metric

      continue

    if quadtree == ds.PKPRQuadTree or quadtree == ds.PKPMRQuadTree:
      if block_A.diameter() > block_B.diameter():
        for child in block_A.children:
          queue.append((block_B, child))
      else:
        for child in block_B.children:
          queue.append((block_A, child))
    else:
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
  
  # plot points
  x = []
  y = []
  for p in points:
    x.append(p.x)
    y.append(p.y)
  #fig = plt.scatter(x, y)
  ax[0].scatter(x, y)
  ax[1].scatter(x, y)

  avg_metric = sum(vals)/len(vals)
  var_metric = sumsq/len(vals) - (avg_metric*avg_metric)

  kurtosis = sum([(x-avg_metric)**4 for x in vals])/(len(vals)*var_metric**2)

  svals = [(x-avg_metric)/var_metric for x in vals]

  mean = sum(svals)/len(svals)

  tailed = (statistics.median(vals) - avg_metric)/(max(vals)-min(vals))

  return rootNode, wsp_count
