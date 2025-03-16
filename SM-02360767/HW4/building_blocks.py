import numpy as np


class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        while True:
            if np.random.rand() < goal_prob:
                return goal_conf
            
            random_conf = np.array([
                np.random.uniform(
                    self.ur_params.mechamical_limits[joint][0], 
                    self.ur_params.mechamical_limits[joint][1]
                )
                for joint in self.ur_params.mechamical_limits
            ])
            if self.config_validity_checker(random_conf):
                return random_conf

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
        sphere_coords = self.transform.conf2sphere_coords(conf)
        for link in self.ur_params.ur_links:
            if sphere_coords[link][0][0] > 0.4:
                return False

        for link1, link2 in self.possible_link_collisions:
            spheres1 = sphere_coords[link1]
            spheres2 = sphere_coords[link2]
            for sphere1 in spheres1:
                for sphere2 in spheres2:
                    dist = np.linalg.norm(sphere1 - sphere2)
                    if dist < (self.ur_params.sphere_radius[link1] + self.ur_params.sphere_radius[link2]):
                        return False

        for link in self.ur_params.ur_links:
            spheres = sphere_coords[link]
            for sphere in spheres:
                for obstacle in self.env.obstacles:
                    dist = np.linalg.norm(sphere - obstacle)
                    if dist < (self.ur_params.sphere_radius[link] + self.env.radius):
                        return False

        return True


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # TODO: HW2 5.2.4
        num_steps = int(np.linalg.norm(current_conf - prev_conf) / self.resolution)
        interpolated_confs = [
            prev_conf + t * (current_conf - prev_conf) / num_steps
            for t in range(1, num_steps + 1)
        ]
        for conf in interpolated_confs:
            if not self.config_validity_checker(conf):
                return False
        return True

    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5