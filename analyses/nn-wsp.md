# Nearest Well Seperated Pair (NN-WSP)

## Introduction
My plan here was to create a new method to solve the TSP where we would jump between well seperated pairs. Unfortunately this algorithm at least in the way I imagined it did not work very well.

## Algorithm
### Subproblem Ordering (takes in starting subtree and tolerance for brute force)
Generate a list of subtrees from the decomposition which contain no greater than `t` points and try to order the subtrees so that they are the nearest well seperated pair to the previous subtree. This is done by:
1. Starting with the starting subtree (traditionally the one with the most points) run the following
    1. Loop through each well seperated pair to the current subtree from closest to furthest and check if that subtree can be added and if it can then do so
    2. If no well seperated pair of the current subtree is a viable candidate then just choose whatever the closest subtree is that is somewhere in the decomposition and shares no points

To check if a subtree can be added do the following:
1. If this subtree is already in the list skip it
2. If any point in this subtree is already in the list skip it (this means that it or one of its children is already in the list and if it is just a child then the siblings should be targeted rather than going to the parent) <!-- REVIEW -->
3. If the subtree contains more than `t` points create another TSP of just the subtree and generate the subproblem order on that tree and add it to the list
4. If the subtree contains less than `t` points add it to the list

### SubProblem Ordering to Point Ordering
Following along the list of subtrees connect the closest points between each subtree and then brute force the shortest path between each entry and exit point of each subtree.

## Results 
| Problem | Error Factor to NN   |
|----------|-----------|
| berlin52 | 1.71355   |
| ch130    | 2.449432  |
| ch150    | 2.639145  |
| eil101   | 1.304851  |
| eil51    | 1.156008  |
| eil76    | 1.425211  |
| kroA100  | 2.090619  |
| kroC100  | 1.462587  |
| kroD100  | 2.49513   |
| lin105   | 1.925139  |
| pcb442   | 5.994045  |
| pr1002   | 12.647018 |
| pr2392   | 14.609548 |
| pr76     | 1.679712  |
| st70     | 1.546154  |
| tsp225   | 2.750139  |

## Code

```
def generate_sub_problem_order(self, start, t) -> list[QuadTreeType]:
        if start is None:
            start = max(self.single_indexable_wspd.keys(), key=len)

        collected_points : set[ds.Point] = set() # track which points we still need
        sub_problem_order : list[QuadTreeType] = [] # order in which to visit subproblems

        def try_to_add(subtree: QuadTreeType) -> bool:
            if subtree in sub_problem_order:
                return False
            elif any(point in collected_points for point in subtree.covered_points):
                return False # TODO: better handling
                # raise NotImplementedError("This should never happen")
            assert all(
                point not in collected_points for point in subtree.covered_points
            ), "This should never happen"
            if len(subtree) <= t:
                collected_points.update(subtree.covered_points)
                sub_problem_order.append(subtree)
                return True
            else:
                sub_tsp = TravellingSalesmanProblem(subtree, np.array([None, None]), self._s, None)
                sub_sub_order = sub_tsp.generate_sub_problem_order(None, t)
                sub_problem_order.extend(sub_sub_order)
                collected_points.update(subtree.covered_points) # REVIEW: Potentially dangerous
                return True
            return False


        current_subtree = start # node is a bad name for this
        assert try_to_add(current_subtree), "Should be guaranteed to insert the first one"
        while len(collected_points) < len(self.points):
            for problem in self.single_indexable_wspd[current_subtree]:
                if try_to_add(problem[0]):
                    current_subtree = problem[0]
                    break
            else:
                # REVIEW: what are we doing here why not just go up?
                jumpable = min(filter(lambda a: all(point not in collected_points for point in a.covered_points), self.single_indexable_wspd.keys()), key=lambda b: (b.center - current_subtree.center).mag())
                assert try_to_add(jumpable), "Should have selected a guaranteed jumpable node"
        return sub_problem_order

    @cache
    def nwsp_path(self, t=8) -> tuple[list[ds.Point], float, tuple]:
        biggest = max(self.single_indexable_wspd.keys(), key=len) # start with the subproblem with most points -> it should be the biggest
        # biggest = max(self.single_indexable_wspd[biggest], key=lambda x: x.radius) # choose the biggest radius ie what must be the most seperated
        # biggest = max(self.single_indexable_wspd.values(), key=lambda x: len(x[1]))[0] # choose the wsp with the most options
        
        sub_problem_order = self.generate_sub_problem_order(biggest, t=t)

        
        # MARK: Connect the subproblems
        tour : list[ds.Point] = []

        start, entry_point = util.min_proj(sub_problem_order[0].covered_points, sub_problem_order[1].covered_points)
        tour.append(start)
        for i in range(1, len(sub_problem_order) - 1):
            exit_point, next_entry = util.min_proj(sub_problem_order[i].covered_points, sub_problem_order[i + 1].covered_points)
            if sub_problem_order[i].leaf: # If we only have one point don't run bfp
                tour.append(sub_problem_order[i].covered_points[0])
                entry_point = next_entry
                continue
            elif entry_point == exit_point: # entry and exit may not be equal, TODO: make this better
                popped = tuple(filter(lambda x: x != entry_point, sub_problem_order[i].covered_points))
                exit_point, next_entry = util.min_proj(popped, sub_problem_order[i + 1].covered_points)
                # TODO: choose whichever new point is less costly
                # alt_prev, alt_entry = util.min_proj(sub_problem_order[i - 1].covered_points, popped)
                # alt_exit, alt_next_entry = util.min_proj(popped, sub_problem_order[i + 1].covered_points)
                # exit_added_cost = None
            tour.extend(util.hamiltonian_path(entry_point, exit_point, sub_problem_order[i].covered_points))
            entry_point = next_entry

        if sub_problem_order[-1].leaf and sub_problem_order[0].leaf:
            tour.extend((entry_point, start))
            return tour, calc_dist(tour), None # TODO: fill in None
        elif sub_problem_order[-1].leaf:
            tour.extend(util.hamiltonian_path(entry_point, start, sub_problem_order[0].covered_points + [entry_point,]))
            return tour, calc_dist(tour), None # TODO: fill in None
        elif sub_problem_order[0].leaf:
            tour.extend(util.hamiltonian_path(entry_point, start, sub_problem_order[-1].covered_points + [start,]))
            return tour, calc_dist(tour), None # TODO: fill in None

        exit_point, next_entry = util.min_proj(sub_problem_order[-1].covered_points, sub_problem_order[0].covered_points)
        if entry_point == exit_point or next_entry == start: # entry and exit may not be equal, TODO: make this better
            popped_end = tuple(filter(lambda x: x != entry_point, sub_problem_order[-1].covered_points)) if entry_point == exit_point else sub_problem_order[-1].covered_points
            popped_start = tuple(filter(lambda x: x != start, sub_problem_order[0].covered_points)) if next_entry == start else sub_problem_order[0].covered_points
            exit_point, next_entry = util.min_proj(popped_end, popped_start)
        tour.extend(util.hamiltonian_path(entry_point, exit_point, sub_problem_order[-1].covered_points))
        tour.extend(util.hamiltonian_path(next_entry, start, sub_problem_order[0].covered_points))

        return tour, calc_dist(tour), None # TODO: fill in None
```

