import numpy as np
from RRTTree import RRTTree
import time

class RRT_STAR(object):
    def __init__(self, bb, ext_mode, goal_prob):
        # Set environment and search parameters
        self.bb = bb
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.eta = 0.2  # Step size for extension mode E2

    def plan(self, start, goal):
        '''
        Compute and return the plan from start to goal.
        Returns a tuple: (np.array(plan), total_cost).
        '''
        start_time = time.time()

        # Update start and goal, and initialize the search tree
        self.start = start
        self.goal = goal
        self.tree = RRTTree(self.bb)
        root_id = self.tree.add_vertex(self.start)  # Add root vertex
        assert root_id == self.tree.get_root_id()  # Ensure it is the root

        while True:
            # Sample a random configuration
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)

            # Find the nearest configuration in the tree
            nearest_id, near_config = self.tree.get_nearest_config(rand_config)

            # Extend towards the random configuration
            new_config = self.extend(near_config, rand_config)

            # Check if the new configuration is valid
            if self.bb.edge_validity_checker(near_config, new_config):
                new_id = self.tree.add_vertex(new_config)
                edge_cost = self.bb.compute_distance(near_config, new_config)
                self.tree.add_edge(nearest_id, new_id, edge_cost=edge_cost)

                # Check if the goal is reached
                if self.bb.compute_distance(new_config, self.goal) < self.eta:
                    goal_id = self.tree.add_vertex(self.goal)
                    self.tree.add_edge(new_id, goal_id, edge_cost=self.bb.compute_distance(new_config, self.goal))
                    break

        # Extract the path from the tree (from start to goal)
        plan = self._extract_plan()
        total_cost = self.compute_cost(plan)
        execution_time = time.time() - start_time
        print(f"Planning completed in {execution_time:.2f} seconds with cost {total_cost}")
        return np.array(plan), total_cost

    def compute_cost(self, plan):
        '''
        Compute and return the cost of the plan, which is the sum of distances between consecutive configurations.
        @param plan: The computed plan for the robot.
        '''
        total_cost = 0
        for i in range(len(plan) - 1):
            total_cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return total_cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration based on the sampled configuration.
        @param near_config: The configuration closest to the sampled configuration.
        @param rand_config: The sampled configuration.
        '''
        if self.ext_mode == "E1":
            # Extend fully to the sampled configuration (E1)
            return rand_config
        elif self.ext_mode == "E2":
            # Extend by a step size eta towards the sampled configuration (E2)
            dist = self.bb.compute_distance(near_config, rand_config)
            if dist <= self.eta:
                return rand_config
            return near_config + self.eta * ((rand_config - near_config) / dist)
        else:
            raise ValueError(f"Invalid extension mode: {self.ext_mode}")

    def _extract_plan(self):
        '''
        Extract the path from the tree, starting from the goal and tracing back to the root.
        '''
        path = []
        current_id = self.tree.get_idx_for_config(self.goal)
        while current_id is not None:
            path.append(self.tree.vertices[current_id].config)
            current_id = self.tree.edges.get(current_id, None)
        path.reverse()
        return path
