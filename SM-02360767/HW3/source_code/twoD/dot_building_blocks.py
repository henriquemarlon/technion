import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # Define dimensionality of the configuration space
        self.dim = 2

        # Robot field of view (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # Visibility distance for the robot's end-effector
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        """
        Euclidean distance between two [x, y] states.
        """
        return np.linalg.norm(np.array(next_config) - np.array(prev_config))

    def sample_random_config(self, goal_prob, goal):
        """
        With probability goal_prob, return the goal; otherwise sample uniformly.
        """
        while True:
            if np.random.rand() < goal_prob:
                sample = goal
            else:
                x = np.random.uniform(self.env.xlimit[0], self.env.xlimit[1])
                y = np.random.uniform(self.env.ylimit[0], self.env.ylimit[1])
                sample = np.array([x, y], dtype=float)

            if self.env.config_validity_checker(sample):
                return sample

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)
