import numpy as np
import heapq

class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        # Track cost-to-come (g) or any other node info
        self.nodes = {}
        
        # Parents for path reconstruction
        self.parents = {}
        
        # For visualization
        self.expanded_nodes = []

        # 8-connected neighborhood
        self.directions = [
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (-1, -1), (1, -1), (-1, 1), (1, 1)
        ]
        self.epsilon = 20.0

    def plan(self):
        """
        Executes Weighted A*: 
        - Return the final path as a list/array of [x, y] states.
        """
        path = self.a_star(self.start, self.goal)
        if path:
            path_array = np.array(path)
            total_cost = self.compute_cost(path_array)
            print(f"Total cost: {total_cost}")
            return path_array
        else:
            print("No path found.")
            return []

    def compute_heuristic(self, state):
        """
        Weighted A* heuristic = epsilon * (Euclidean distance to goal).
        """
        return self.epsilon * self.bb.compute_distance(state, self.goal)

    def a_star(self, start_loc, goal_loc):
        """
        Standard A* loop with weighted heuristic f(n)=g(n)+h(n).
        """
        open_set = []
        heapq.heapify(open_set)

        g_values = {}
        visited = set()

        g_values[tuple(start_loc)] = 0.0
        f_start = self.compute_heuristic(start_loc)
        heapq.heappush(open_set, (f_start, tuple(start_loc)))
        self.parents[tuple(start_loc)] = None

        while open_set:
            _, current_node = heapq.heappop(open_set)
            if current_node in visited:
                continue
            visited.add(current_node)
            self.expanded_nodes.append(np.array(current_node))

            if np.allclose(np.array(current_node), goal_loc):
                return self.reconstruct_path(current_node)

            for dx, dy in self.directions:
                neighbor = (current_node[0] + dx, current_node[1] + dy)
                if neighbor not in visited:
                    neighbor_arr = np.array(neighbor, dtype=float)
                    if (self.bb.config_validity_checker(neighbor_arr) and
                        self.bb.edge_validity_checker(np.array(current_node), neighbor_arr)):

                        step_cost = self.bb.compute_distance(current_node, neighbor)
                        tentative_g = g_values[current_node] + step_cost
                        
                        if (neighbor not in g_values) or (tentative_g < g_values[neighbor]):
                            g_values[neighbor] = tentative_g
                            self.parents[neighbor] = current_node
                            f_val = tentative_g + self.compute_heuristic(neighbor)
                            heapq.heappush(open_set, (f_val, neighbor))

        print("A* did not find a path to the goal.")
        return []

    def reconstruct_path(self, current_node):
        """
        Backtrack from goal to start using parent pointers.
        """
        path = []
        while current_node is not None:
            path.append(list(current_node))
            current_node = self.parents[current_node]
        path.reverse()
        return path

    def get_expanded_nodes(self):
        return self.expanded_nodes

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan: A given plan for the robot.
        '''
        total_cost = 0
        for i in range(len(plan) - 1):
            total_cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return total_cost