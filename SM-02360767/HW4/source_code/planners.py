import numpy as np
from RRTTree import RRTTree
import time

MAX_ITER = 10000000000

class RRT_STAR(object):

    def __init__(self, bb, max_step_size,
                 max_itr=None, stop_on_goal=None, k=None, goal_prob=0.01):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.goal_prob = goal_prob
        self.k = k

        self.step_size = max_step_size

    def find_path(self, start, goal):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        
        self.start = np.array(start)
        self.goal = np.array(goal)
        
        initial_time = time.time()
        
        self.tree.add_vertex(np.array(self.start))
        iterations = MAX_ITER if self.max_itr is None else self.max_itr
        
        for i in range(iterations):
            if i % 1000 == 0:
                print(f"Planning iteration {i}")
                
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)
            x_nearest_id, x_nearest = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_nearest, x_rand)
            if self.bb.config_validity_checker(x_new) == False or self.tree.is_goal_exists(x_new):
                continue

            if self.bb.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.tree.add_vertex(np.array(x_new))
                edge_cost = self.bb.compute_distance(x_nearest, x_new)
                self.tree.add_edge(x_nearest_id, x_new_id, edge_cost)

                X_near_id, X_near = self.tree.get_k_nearest_neighbors(x_new, self.k)

                for x_near_id, x_near in zip(X_near_id, X_near):
                    self.rewire_rrt_star(x_near, x_near_id, x_new, x_new_id)

                for x_near_id, x_near in zip(X_near_id, X_near):
                    self.rewire_rrt_star(x_new, x_new_id, x_near, x_near_id)

            if self.stop_on_goal and self.tree.is_goal_exists(self.goal):
                print(f"Goal found at iteration {i}")
                break
        
        if i >= iterations - 1:
            print("Max iterations reached")
        
        plan = self.return_path()
        if plan is None:
            print("No valid path found.")
            return None, None

        cost = self.compute_cost(plan)
        
        print(f"Time taken: {time.time() - initial_time:.2f} seconds")
        return plan, cost
    
    def rewire_rrt_star(self, x_near, x_near_id, x_new, x_new_id):

        edge_cost = self.bb.compute_distance(x_near, x_new)
        new_cost = self.tree.vertices[x_near_id].cost + edge_cost

        if new_cost < self.tree.vertices[x_new_id].cost and self.bb.edge_validity_checker(x_near, x_new):
            if x_new_id in self.tree.edges:
                del self.tree.edges[x_new_id]

            self.tree.add_edge(x_near_id, x_new_id, edge_cost)
            self.update_vertex_costs(x_new_id)

    def update_vertex_costs(self, vertex_id):
        queue = [vertex_id]
        edges = self.tree.get_edges_as_states()
        while queue:
            current_id = queue.pop(0)
            childern_configs = [state[1] for state in edges if state[0].any() == current_id]
            children_ids = [self.tree.get_idx_for_config(config) for config in childern_configs]

            current_cost = self.tree.vertices[current_id].cost
            edges_cost = [self.bb.compute_distance(
                self.tree.vertices[current_id].config, child_config
            ) for child_config in childern_configs]

            for child_id, edge_cost in zip(children_ids, edges_cost):
                if child_id is not None:
                    new_cost = current_cost + edge_cost
                    if new_cost < self.tree.vertices[child_id].cost:
                        self.tree.vertices[child_id].set_cost(new_cost)
                        queue.append(child_id)
    
    def compute_cost(self, plan):
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return float(cost)

    def extend(self, x_near, x_rand):
        if self.step_size is None:
            return x_rand
        direction = x_rand - x_near
        length = np.linalg.norm(direction)
        if length > self.step_size:
            direction = direction / length * self.step_size
        return x_near + direction
    
    def return_path(self):
        path = []
        curr_state = self.goal
        while(curr_state is not None):
            path.append(curr_state)
            if(curr_state == self.start).all():
                break
            curr_state_id = self.tree.get_idx_for_config(curr_state)
            if curr_state_id is None:
                return None
            try:
                next_state_index = self.tree.edges[curr_state_id]
                curr_state = self.tree.vertices[next_state_index].config
            except KeyError:
                # Handle case where current state doesn't have an edge
                return None

        path.reverse()
        return np.array(path)