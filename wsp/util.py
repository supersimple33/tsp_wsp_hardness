import math
from wsp import ds

def euclidDist(p1, p2):
    return math.sqrt(((p2.x - p1.x) ** 2) + ((p2.y - p1.y) ** 2))

def calcDist(points):
    dist = 0
    for i in range(len(points) - 1):
        dist += euclidDist(points[i], points[i+1])
    return dist

def sublist_get_points(lst):
    points = []
    for item in lst:
        if isinstance(item, list):
            points += sublist_get_points(item)
        else:
            points.append(item)
    return points

def min_proj_set_or_point(item_A, item_B, min2=False):
    if not isinstance(item_A, list):
        item_A = [item_A]
    if not isinstance(item_B, list):
        item_B = [item_B]
    item_A = sublist_get_points(item_A)
    item_B = sublist_get_points(item_B)

    if min2:
        return min2_proj(item_A, item_B)
    return min_proj(item_A, item_B)

def min_proj(set_A, set_B):
    """Min pair between points from set_A and set_B"""
    '''avg_A = ds.Point(0,0)
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
            min_d2 = dist'''
    mind = 99999999
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

def min2_proj(set_A, set_B):
    """Min pair between points from set_A and set_B"""
    avg_A = ds.Point(0,0)
    avg_B = ds.Point(0,0)
    for p_A in set_A:
        avg_A += p_A
    for p_B in set_B:
        avg_B += p_B
    avg_A /= len(set_A)
    avg_B /= len(set_B)

    '''min_d1 = 99999999
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
            sec_p2 = p_B'''

    mind = 99999999
    min_p1 = None
    min_p2 = None
    sec_p1 = None
    sec_p2 = None
    rank = []
    #print(set_A, set_B)
    for p_A in set_A:
        for p_B in set_B:
            dist = p_A.distance_to(p_B)
            inserted = False
            if len(rank) > 0:
                for i in range(len(rank)):
                    tup = rank[i]
                    if dist < tup[2]:
                        rank.insert(i, (p_A, p_B, dist)) 
                        inserted = True
                        break
            if not inserted:
                rank.append((p_A, p_B, dist))


            '''if dist < mind:
                mind = dist
                if min_p1 != None and avg_B.distance_to(p_A) < avg_B.distance_to(min_p1): #p_A != min_p1:
                    sec_p1 = min_p1
                if min_p2 != None and avg_A.distance_to(p_B) < avg_A.distance_to(min_p2): #p_B != min_p2:
                    sec_p2 = min_p2
                min_p1 = p_A
                min_p2 = p_B'''
    min_p1 = rank[0][0]
    min_p2 = rank[0][1]
    i = 1
    while i < len(rank) and (sec_p1 == None or sec_p2 == None):
        #print("searching for", i, rank[i])
        if sec_p1 == None and rank[i][0] != min_p1:
            sec_p1 = rank[i][0]
        if sec_p2 == None and rank[i][1] != min_p2:
            sec_p2 = rank[i][1]
        i += 1
    #print("min proj 2", min_p1, min_p2, sec_p1, sec_p2)
    return min_p1, min_p2, sec_p1, sec_p2