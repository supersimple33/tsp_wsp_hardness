import numpy as np
import numba as nb
from numba import njit, types
from numba.typed import List

# NOTE: the id of leaves is 0 to support unsigned int types and it is known that the root is always 0 and it cannot be a kid

LEAF_ID = 0

@njit(inline='always')
def _build_fair_split_tree[float_dtype: np.floating](children: np.ndarray, node_ranges: np.ndarray, points: np.ndarray[tuple[int, int], np.dtype[float_dtype]], int_dtype: type[np.integer]):
    """
    Builds an array-based binary split tree for the input points.
    Splits nodes spatially along their longest bounding-box axis.
    """
    n, d = points.shape
    max_nodes = 2 * n - 1
    
    # Tree data structures # TODO: update init on a better numab version
    centers = np.empty((max_nodes, d), dtype=points.dtype)
    radii = np.empty(max_nodes, dtype=points.dtype)

    indices = np.arange(n, dtype=int_dtype)
    
    # Initialize root
    node_ranges[0]['start'] = 0
    node_ranges[0]['end'] = n
    node_count = 1
    
    # Use a standard list as a stack for iterative traversal (Numba handles this perfectly)
    stack = [0]
    
    bmin = np.empty(d, dtype=points.dtype)
    bmax = np.empty(d, dtype=points.dtype)
    while len(stack) > 0:
        u = stack.pop()
        start = node_ranges[u]['start']
        end = node_ranges[u]['end']

        # Base case: Leaf node
        if end - start <= 1:
            children[u]['left'] = LEAF_ID
            children[u]['right'] = LEAF_ID
            centers[u] = points[indices[start]]
            radii[u] = 0.0
            continue
        
        # 1. Compute Bounding Box
        bmin[:] = points[indices[start]]
        bmax[:] = points[indices[start]]
        for i in range(start + 1, end):
            idx = indices[i]
            for j in range(d):
                if points[idx, j] < bmin[j]: bmin[j] = points[idx, j]
                if points[idx, j] > bmax[j]: bmax[j] = points[idx, j]
        
        # 2. Store Center and Radius (circumradius of bounding box)
        for j in range(d): # centers[u] = (bmin + bmax) / 2.0 but no memory alloc
            centers[u, j] = (bmin[j] + bmax[j]) / 2.0
            
        dist_sq = (bmax[0] - bmin[0]) ** 2
        for j in range(1, d): # dist_sq = np.sum((bmax - bmin) ** 2) but no memory alloc
            dist_sq += (bmax[j] - bmin[j]) ** 2
        radii[u] = np.sqrt(dist_sq) / 2.0
            
        # 3. Find longest axis to split
        axis = 0
        max_len = bmax[0] - bmin[0]
        for j in range(1, d):
            if bmax[j] - bmin[j] > max_len:
                max_len = bmax[j] - bmin[j]
                axis = j
        
        # 4. Partition points around the spatial midpoint
        mid_val = (bmin[axis] + bmax[axis]) / 2.0
        left_ptr = start
        right_ptr = end - 1
        
        while left_ptr <= right_ptr:
            if points[indices[left_ptr], axis] <= mid_val:
                left_ptr += 1
            else:
                # Swap elements
                temp = indices[left_ptr]
                indices[left_ptr] = indices[right_ptr]
                indices[right_ptr] = temp
                right_ptr -= 1
        
        mid = left_ptr
        # Fallback to median split if points are identical along the longest axis
        if mid == start or mid == end:
            mid = start + (end - start) // 2
            
        # 5. Create children
        children[u]['left'] = node_count
        node_ranges[node_count]['start'] = start
        node_ranges[node_count]['end'] = mid
        stack.append(node_count)
        node_count += 1
        
        children[u]['right'] = node_count
        node_ranges[node_count]['start'] = mid
        node_ranges[node_count]['end'] = end
        stack.append(node_count)
        node_count += 1
        
    return centers, radii, indices

@njit
def _find_wspd(u, v, s, children, centers, radii, pairs):
    """
    Recursively compares two nodes. If they are well-separated, adds them 
    to the pair list. Otherwise, splits the larger node.
    """
    dist_sq = (centers[u, 0] - centers[v, 0]) ** 2
    for j in range(1, centers.shape[1]):
        dist_sq += (centers[u, j] - centers[v, j]) ** 2
        
    r_u = radii[u]
    r_v = radii[v]
    
    # Separation bound checking (using squared distance to avoid sqrt overhead)
    threshold = (2 * s * max(r_u, r_v)) + r_u + r_v
    
    if dist_sq >= threshold * threshold:
        pairs.append((u, v))
        return
        
    # Not well separated: Split the node with the larger radius
    if r_u == r_v == 0.0:
        # each group is of the same single point and going to smaller levels will not help any further
        return
    
    if r_u >= r_v:
        _find_wspd(children[u]['left'], v, s, children, centers, radii, pairs)
        _find_wspd(children[u]['right'], v, s, children, centers, radii, pairs)
    else:
        _find_wspd(u, children[v]['left'], s, children, centers, radii, pairs)
        _find_wspd(u, children[v]['right'], s, children, centers, radii, pairs)

@njit
def _traverse_and_wspd(node, s, children, centers, radii, pairs):
    """
    Traverses the tree to generate all internal WSPD pairs.
    """
    if children[node]['left'] == LEAF_ID: return
    
    # Recurse into children
    _traverse_and_wspd(children[node]['left'], s, children, centers, radii, pairs)
    _traverse_and_wspd(children[node]['right'], s, children, centers, radii, pairs)
    
    # Find cross pairs between left and right children
    _find_wspd(children[node]['left'], children[node]['right'], s, children, centers, radii, pairs)

#@njit # TODO: make this njit cause why not in a future version
def get_wspd(points: np.ndarray, s: float, int_dtype: type[np.signedinteger] = np.int64):
    """
    Main entry point. Builds the tree and computes the WSPD.
    """
    assert len(points) >= 2, "At least two points are required to compute a WSPD."

    max_nodes = 2 * len(points) - 1
    children = np.empty(max_nodes, dtype=[('left', int_dtype), ('right', int_dtype)])
    node_ranges = np.empty(max_nodes, dtype=[('start', int_dtype), ('end', int_dtype)])
        
    centers, radii, indices = _build_fair_split_tree(children, node_ranges, points, int_dtype)
    
    pairs = List.empty_list(nb.types.Tuple((nb.from_dtype(int_dtype), nb.from_dtype(int_dtype)))) # TODO: preallocate this list
    
    _traverse_and_wspd(0, s, children, centers, radii, pairs)
    
    return pairs, node_ranges, indices


if __name__ == "__main__":
    get_wspd(np.array([[0.0, 0.0], [1.0, 1.1]]), 1.5)
