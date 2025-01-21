import numpy as np
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):
    def __init__(self, bb, start, ext_mode, goal_prob, coverage):
        # Set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # Set search parameters
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        # Set step size
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()
        root_id = self.tree.add_vertex(self.start, inspected_points=np.empty((0, 2)))  # Start with empty inspected points
        assert root_id == self.tree.get_root_id()

        while self.tree.max_coverage < self.coverage:
            # Sample a random configuration
            rand_config = self.bb.sample_random_config(self.goal_prob, self.start)

            # Find the nearest configuration in the tree
            nearest_id, near_config = self.tree.get_nearest_config(rand_config)

            # Extend towards the random configuration
            new_config = self.extend(near_config, rand_config)

            # Check if the new configuration is valid
            if self.bb.edge_validity_checker(near_config, new_config):
                # Compute the union of inspected points
                new_inspected_points = self.bb.compute_union_of_points(
                    self.tree.vertices[nearest_id].inspected_points,
                    self.bb.get_inspected_points(new_config),
                )
                new_id = self.tree.add_vertex(new_config, inspected_points=new_inspected_points)
                edge_cost = self.bb.compute_distance(near_config, new_config)
                self.tree.add_edge(nearest_id, new_id, edge_cost=edge_cost)

        # Retrieve the path from the best coverage vertex to the root
        best_vertex_id = self.tree.max_coverage_id
        plan = self._extract_plan(best_vertex_id)
        execution_time = time.time() - start_time
        print(f"Planning completed in {execution_time:.2f} seconds with coverage {self.tree.max_coverage:.2f}.")
        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        total_cost = 0
        for i in range(len(plan) - 1):
            total_cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return total_cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        dist = self.bb.compute_distance(near_config, rand_config)
        if dist <= self.step_size:
            return rand_config
        return near_config + self.step_size * ((rand_config - near_config) / dist)

    def _extract_plan(self, vertex_id):
        '''
        Extract the path from the tree, starting from the given vertex and tracing back to the root.
        '''
        path = []
        current_id = vertex_id
        while current_id is not None:
            path.append(self.tree.vertices[current_id].config)
            current_id = self.tree.edges.get(current_id, None)
        path.reverse()
        return path
