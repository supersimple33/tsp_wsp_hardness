# Well Separated Pairs and the Traveling Salesman Problem

### Instructions for Running
>python tsp-nnp.py \<TSP dataset file\> <separation factor> <quadtree:{-p, -pr, -pmr, -pkpr, -pkpmr}> <flags:{-b<int>, -d, -bf}> \
>python tsp-nnp.py att48.tsp 5 \
>python tsp-sps.py att48.tsp 1 -pr -b4 \
>python tsp-sps.py vlsi/xqf131.tsp 2 -pkpr \
>python tsp-spop.py vlsi/xqf131.tsp 1 -pkpmr       <--- Recommended \
>python tsp-treesp.py vlsi/xqf131.tsp 1 -pkpmr     <--- Recommended \
>python wsp-quadtree-test.py att48.tsp 1 -pr -b4 \
>python wsp-quadtree-test.py att48.tsp 1 -pmr

```
quadtrees:
-p: point quadtree 
-pr: point region quadtree
-pmr: PMR quadtree 
-pkpr: PK PR quadtree 
-pkpmr: PK PMR quadtree

flags:
-b<int>: bucket size (for pr and pmr quadtrees), defaults to -b1, ex: -b4
-s: block shrinking on
-d: debug info quadtree and WSP
-bf: brute force, turns off WSP for testing complexity without optimization
```

### Directory Structure
`tsp-bfp.py` - Brute Force + WSP Pruning algorithm - Exponential \
`tsp-nnp.py` - Nearest Neighbor - Polynomial \
`tsp-sps.py` - WSP Subproblem Sorting + Nearest Neighbor (try spop and treesp as they are better) - Polynomial \
`tsp-spop.py` - WSP Subproblem Connection + Brute Force + Nearest Neighbor - Polynomial \
`tsp-treesp.py` - Quadtree Subproblem Connection + Brute Force + Nearest Neighbor - Polynomial \
`tsp-hardness.py` - Compute heuristic to help understand 'hardness' of groups of subproblems to help identify optimal approximation algorithm \
`tsp-wgmst.py` - WSP Graph MST, builds MST out of graph found from WSP min projections - Motivation: https://www.cs.umd.edu/~mount/Indep/Chaojun_Li/final-rept.pdf \
`tsp-msto.py` - WSP Graph MST only - incomplete optimization approach \
`wsp-quadtree-test.py` - Runs WSP alg only to test on different datastructures/quadtrees \
Point set data is located inside the `/data` directory \
WSP implementation inside `/wsp` directory

### Point data
We use the .TSP format (with NODE_COORD_SECTION only) common among TSP datasets.
