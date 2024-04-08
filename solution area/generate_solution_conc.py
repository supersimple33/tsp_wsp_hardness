# NOTE: I must live in this directory since I spit out a lot of files

import time
import glob
import pathlib

import os

import tsplib95
from concorde.tsp import TSPSolver

old_dir = os.getcwd()

os.chdir(pathlib.Path(__file__).parent.resolve()) # Change to my directory so I spit out files here

problems = glob.glob("../ALL_tsp/Tnm*.tsp")
print("found", len(problems), "problems")

for problem in problems: # TODO: add some multiprocessing
    # Load up the problem
    solver = TSPSolver.from_tspfile(problem)
    problem = problem[11:-4] # Chop off the "../ALL_tsp/" and the ".tsp"
    if solver._ncount < 133 or solver._ncount > 133: # Check that we can handle this problem
        # print("Skipping", problem, "because it's too big")
        continue

    print("Starting", problem)
    start = time.time()
    
    solution = solver.solve(verbose=False, random_seed=0) # Uncomment for debugging
    print("Solved", problem, "in", time.time() - start, "seconds")
    
    assert solution.success # Check that the solution is optimal
    assert solution.found_tour # DEBUG: If this fails a different seed should be tried
    
    tsplib95.models.StandardProblem(
            name=f"{problem}", 
            comment=f"Recalculated optimal tour for {problem}, using concorde got a tour of length {solution.optimal_value}",
            type="TOUR", 
            dimension=solver._ncount, 
            tours=[[x + 1 for x in list(solution.tour)],]
        ).save(f"{problem}.opt.tour")
