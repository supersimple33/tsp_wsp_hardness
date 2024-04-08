import time
import glob
import pathlib

import os

import tsplib95
import numpy as np

from wsp import tsp
from wsp import ds
from wsp import file_load

old_dir = os.getcwd()

os.chdir(pathlib.Path(__file__).parent.resolve()) # Change to my directory so I spit out files here

problems = glob.glob("ALL_tsp/Tnm*.tsp")
print("found", len(problems), "problems")

for problem in problems: # TODO: add some multiprocessing
    # Load up the problem
    points = file_load.load_points(problem, False)
    
    problem = problem[8:-4] # Chop off the "../ALL_tsp/" and the ".tsp"
    if len(points) != 133: # Check that we can handle this problem
        # print("Skipping", problem, "because it's too big")
        continue

    # Create the instance
    ts_problem = tsp.TravellingSalesmanProblem[ds.PKPRQuadTree](ds.PKPRQuadTree, points, np.array([None, None]), s=2.0)
    
    # Solve the problem
    print("Starting", problem)
    start = time.time()
    path, length, _ = ts_problem.rtr_path()
    print(f"Calculated a tour of length {length} in {time.time() - start} seconds")
    
    # Should we save the tour?
    # good = input("Is this the optimal solution? (y/n)")
    
    i = 1
    while int(length) != 1944317: # good.lower() != 'y': 
        # Solve the problem
        print("Starting Again On", problem)
        start = time.time()
        path, length, _ = ts_problem.rtr_path(path=path, seed=ts_problem.number_seed + i)
        print(f"Calculated a tour of length {length} in {time.time() - start} seconds on iteration {i}")
        
        # Should we save the tour?
        i += 1
        # good = input("Is this the optimal solution (or close)? (y/n)")
    
    # Local Searching
    print("Local Searching", problem)
    start = time.time()
    path, length, _ = ts_problem.local_search_path(path=path)
    print(f"Calculated a tour of length {length} in {time.time() - start} seconds with LS")
    
    num_tour = ts_problem.point_tour_to_ids(path, offset_add=1)[:-1]
    tsplib95.models.StandardProblem(
        name=f"{problem}", 
        comment=f"Recalculated optimal tour for {problem}, using LKH got a tour of length {length}",
        type="TOUR", 
        dimension=len(points), 
        tours=[num_tour]
    ).save(f"solution area/{problem}.opt.tour")

