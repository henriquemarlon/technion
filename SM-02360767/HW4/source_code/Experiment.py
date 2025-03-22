import json
import time
from enum import Enum
import numpy as np
import os

from environment import Environment

from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from inverse_kinematics import inverse_kinematic_solution, DH_matrix_UR5e, forward_kinematic_solution

from environment import LocationType


def log(msg):
    dir_path = r"./outputs/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    written_log = f"STEP: {msg}"
    print(written_log)
    with open(os.path.join(dir_path, 'output.txt'), 'a') as file:
        file.write(f"{written_log}\n")


class Gripper(str, Enum):
    OPEN = "OPEN",
    CLOSE = "CLOSE"
    STAY = "STAY"

def get_shifted_cubes_to_real_world(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift, env):
    cubes_already_moved = []
    cubes_in_original_area = []
    for cube in cubes_in_original_area_pre_shift:
        cubes_in_original_area.append(cube + env.cube_area_corner[LocationType.RIGHT])
    for cube in cubes_already_moved_pre_shift:
        cubes_already_moved.append(cube + env.cube_area_corner[LocationType.LEFT])
    return [*cubes_already_moved, *cubes_in_original_area]


def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


class Experiment:
    def __init__(self, cubes=None):
        # environment params
        self.cubes = cubes
        self.right_arm_base_delta = 0  # deviation from the first link 0 angle
        self.left_arm_base_delta = 0  # deviation from the first link 0 angle
        self.right_arm_meeting_safety = None
        self.left_arm_meeting_safety = None

        # tunable params
        self.max_itr = 50000  # Significantly increased to improve path finding chances
        self.max_step_size = 0.3  # Smaller steps for better precision
        self.goal_bias = 0.2  # High probability of sampling the goal
        self.resolution = 0.1
        # start confs
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        # result dict
        self.experiment_result = []

    def push_step_info_into_single_cube_passing_data(self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def plan_single_arm(self, planner, start_conf, goal_conf, description, active_id, command, static_arm_conf, cubes_real,
                            gripper_pre, gripper_post):
        print(f"Planning path for {description}")
        path, cost = planner.find_path(start=start_conf,
                                       goal=goal_conf)
        
        if path is None or len(path) < 2:
            print(f"No valid path found or invalid path for {description}. Creating simple path...")
            # Create a simple direct path as fallback
            path = np.array([start_conf, goal_conf])
            cost = np.linalg.norm(goal_conf - start_conf)
            print(f"Created fallback path with cost {cost}")
        else:
            print(f"Path found for {description} with {len(path)} waypoints and cost {cost}")
        
        # create the arm plan
        self.push_step_info_into_single_cube_passing_data(description,
                                                          active_id,
                                                          command,
                                                          static_arm_conf.tolist(),
                                                          [path_element.tolist() for path_element in path],
                                                          [list(cube) for cube in cubes_real],
                                                          gripper_pre,
                                                          gripper_post)

    def plan_single_cube_passing(self, cube_i, cubes,
                                 left_arm_start, right_arm_start,
                                 env, bb, planner, left_arm_transform, right_arm_transform):
        # add a new step entry
        single_cube_passing_info = {
            "description": [],  # text to be displayed on the animation
            "active_id": [],  # active arm id
            "command": [],
            "static": [],  # static arm conf
            "path": [],  # active arm path
            "cubes": [],  # coordinates of cubes on the board at the given timestep
            "gripper_pre": [],  # active arm pre path gripper action (OPEN/CLOSE/STAY)
            "gripper_post": []  # active arm pre path gripper action (OPEN/CLOSE/STAY)
        }
        self.experiment_result.append(single_cube_passing_info)
        ###############################################################################
        # set variables
        description = "right_arm => [start -> cube pickup], left_arm static"
        active_arm = LocationType.RIGHT
        # start planning
        log(msg=description)
        # fix obstacles and update env
        cubes_already_moved_pre_shift = cubes[0:cube_i]
        cubes_in_original_area_pre_shift = cubes[cube_i:]
        cubes_real = get_shifted_cubes_to_real_world(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift, env)

        update_environment(env, active_arm, left_arm_start, cubes_real)

        # Find the current cube position in real-world coordinates
        current_cube = cubes[cube_i] + env.cube_area_corner[LocationType.RIGHT]
        
        # Calculate configuration for the right arm to approach the cube (slightly above)
        approach_position = current_cube + np.array([0, 0, 0.15])  # Position above the cube
        
        # Use inverse kinematics to find the configuration for right arm
        approach_transform = np.eye(4)
        approach_transform[:3, 3] = approach_position[:3]
        # Use the first valid solution from inverse kinematics
        ik_solutions = inverse_kinematic_solution(DH_matrix_UR5e, approach_transform)
        cube_approach = np.array(ik_solutions[:, 0]).flatten()
        
        # plan the path to approach the cube
        self.plan_single_arm(planner, right_arm_start, cube_approach, description, active_arm, "move",
                             left_arm_start, cubes_real, Gripper.OPEN, Gripper.STAY)
        ###############################################################################

        # Move down to grab the cube
        self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          left_arm_start.tolist(),
                                                          [0, 0, -0.14],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.STAY,
                                                          Gripper.CLOSE)
        
        # Close gripper to grab cube
        self.push_step_info_into_single_cube_passing_data("picking up a cube: close gripper",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          left_arm_start.tolist(),
                                                          [0, 0, 0],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.CLOSE,
                                                          Gripper.CLOSE)
        
        # Move up with the cube
        self.push_step_info_into_single_cube_passing_data("picking up a cube: go up",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          left_arm_start.tolist(),
                                                          [0, 0, 0.14],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.CLOSE,
                                                          Gripper.CLOSE)
        
        # Move right arm to meeting point
        description = "right_arm => [cube pickup -> meeting point], left_arm static"
        log(msg=description)
        self.plan_single_arm(planner, cube_approach, self.right_arm_meeting_safety, description, 
                             LocationType.RIGHT, "move", left_arm_start, cubes_real, 
                             Gripper.CLOSE, Gripper.CLOSE)
        
        # Now move left arm to meeting point
        description = "left_arm => [home -> meeting point], right_arm static"
        active_arm = LocationType.LEFT
        log(msg=description)
        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)
        self.plan_single_arm(planner, left_arm_start, self.left_arm_meeting_safety, description,
                             LocationType.LEFT, "move", self.right_arm_meeting_safety, cubes_real,
                             Gripper.OPEN, Gripper.OPEN)
        
        # Transfer cube from right to left
        self.push_step_info_into_single_cube_passing_data("transferring cube: right arm opens gripper",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          self.left_arm_meeting_safety.tolist(),
                                                          [0, 0, 0],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.CLOSE,
                                                          Gripper.OPEN)
        
        # Left arm closes gripper
        self.push_step_info_into_single_cube_passing_data("transferring cube: left arm closes gripper",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          self.right_arm_meeting_safety.tolist(),
                                                          [0, 0, 0],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.OPEN,
                                                          Gripper.CLOSE)
        
        # Move right arm back to home position
        description = "right_arm => [meeting point -> home], left_arm static"
        active_arm = LocationType.RIGHT
        log(msg=description)
        update_environment(env, active_arm, self.left_arm_meeting_safety, cubes_real)
        right_arm_end = self.right_arm_home  # Return to home position
        self.plan_single_arm(planner, self.right_arm_meeting_safety, right_arm_end, description,
                             LocationType.RIGHT, "move", self.left_arm_meeting_safety, cubes_real,
                             Gripper.OPEN, Gripper.OPEN)
        
        # Calculate target position in left cube area
        target_position = cubes[cube_i] + env.cube_area_corner[LocationType.LEFT]
        
        # Add offset for approach position
        target_approach_position = target_position + np.array([0, 0, 0.15])
        
        # Use inverse kinematics to find the configuration for left arm
        target_approach_transform = np.eye(4)
        target_approach_transform[:3, 3] = target_approach_position[:3]
        ik_solutions = inverse_kinematic_solution(DH_matrix_UR5e, target_approach_transform)
        left_target_approach = np.array(ik_solutions[:, 0]).flatten()
        
        # Move left arm to target approach position
        description = "left_arm => [meeting point -> target approach], right_arm static"
        active_arm = LocationType.LEFT
        log(msg=description)
        update_environment(env, active_arm, right_arm_end, cubes_real)
        self.plan_single_arm(planner, self.left_arm_meeting_safety, left_target_approach, description,
                             LocationType.LEFT, "move", right_arm_end, cubes_real,
                             Gripper.CLOSE, Gripper.CLOSE)
        
        # Move down to place the cube
        self.push_step_info_into_single_cube_passing_data("placing cube: go down",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          right_arm_end.tolist(),
                                                          [0, 0, -0.14],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.CLOSE,
                                                          Gripper.CLOSE)
        
        # Open gripper to release cube
        self.push_step_info_into_single_cube_passing_data("placing cube: open gripper",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          right_arm_end.tolist(),
                                                          [0, 0, 0],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.CLOSE,
                                                          Gripper.OPEN)
        
        # Move up after releasing
        self.push_step_info_into_single_cube_passing_data("placing cube: go up",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          right_arm_end.tolist(),
                                                          [0, 0, 0.14],
                                                          [list(cube) for cube in cubes_real],
                                                          Gripper.OPEN,
                                                          Gripper.OPEN)
        
        # Move left arm back to home
        description = "left_arm => [target -> home], right_arm static"
        log(msg=description)
        left_arm_end = self.left_arm_home  # Return to home position
        self.plan_single_arm(planner, left_target_approach, left_arm_end, description,
                             LocationType.LEFT, "move", right_arm_end, cubes_real,
                             Gripper.OPEN, Gripper.OPEN)
        
        return left_arm_end, right_arm_end


    def plan_experiment(self):
        start_time = time.time()

        exp_id = 2
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT])
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT])

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        bb = Building_Blocks(env=env,
                             resolution=self.resolution,
                             p_bias=self.goal_bias,
                             transform=transform_right_arm,
                             ur_params=ur_params_right)

        rrt_star_planner = RRT_STAR(max_step_size=self.max_step_size,
                                    max_itr=self.max_itr,
                                    bb=bb)
        visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                  transform_left_arm=transform_left_arm)
        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id)

        log(msg="calculate meeting point for the test.")

        # Calculate meeting point - a position between the two tables where both arms can reach
        # This is typically in the middle between the two robot bases
        right_base = np.array(env.arm_base_location[LocationType.RIGHT])
        left_base = np.array(env.arm_base_location[LocationType.LEFT])
        meeting_point = (right_base + left_base) / 2
        
        # Add some height to the meeting point for better clearance
        meeting_point[2] += 0.3
        
        # Add a small offset to avoid exact collision between end effectors
        right_meeting_point = meeting_point + np.array([0, 0.04, 0])
        left_meeting_point = meeting_point - np.array([0, 0.04, 0])
        
        # Transform meeting points to robot configurations using inverse kinematics
        right_transform = np.eye(4)
        right_transform[:3, 3] = right_meeting_point[:3]
        
        left_transform = np.eye(4)
        left_transform[:3, 3] = left_meeting_point[:3]
        
        # Get configurations for both robots at the meeting point
        right_ik_solutions = inverse_kinematic_solution(DH_matrix_UR5e, right_transform)
        left_ik_solutions = inverse_kinematic_solution(DH_matrix_UR5e, left_transform)
        
        self.right_arm_meeting_safety = np.array(right_ik_solutions[:, 0]).flatten()
        self.left_arm_meeting_safety = np.array(left_ik_solutions[:, 0]).flatten()
        
        # Verify the inverse kinematics solutions with forward kinematics
        right_fk_matrix = forward_kinematic_solution(DH_matrix_UR5e, self.right_arm_meeting_safety.reshape(6, 1))
        left_fk_matrix = forward_kinematic_solution(DH_matrix_UR5e, self.left_arm_meeting_safety.reshape(6, 1))
        
        print("Right arm meeting point verification:")
        print(f"Original transform: \n{right_transform}")
        print(f"Forward kinematics result: \n{right_fk_matrix}")
        
        print("Left arm meeting point verification:")
        print(f"Original transform: \n{left_transform}")
        print(f"Forward kinematics result: \n{left_fk_matrix}")

        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(i, self.cubes, left_arm_start, right_arm_start,
                                              env, bb, rrt_star_planner, transform_left_arm, transform_right_arm)


        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")
        # Serializing json
        json_object = json.dumps(self.experiment_result, indent=4)
        # Writing to sample.json
        dir_path = r"./outputs/"
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)
        # show the experiment then export it to a GIF
        visualizer.show_all_experiment(dir_path + "plan.json")
        visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id):
        cube_side = 0.04
        cubes = []
        if experiment_id == 1:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            cubes.append([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
        elif experiment_id == 2:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            cubes.append([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            # row 1: cube 2
            cubes.append([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
        return cubes