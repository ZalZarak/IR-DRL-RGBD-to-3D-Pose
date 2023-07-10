from abc import ABC

import numpy as np

from modular_drl_env.world import World
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.shapes import Box


class PeterWorld(World):
    def __init__(self, sim_step: float, sim_steps_per_env_step: int, env_id: int, assets_path: str, target_size_max: float, target_size_min: float,
                 target_size_divisor: float, target_size_success_threshold: float):
        super().__init__([-2, 2, -2, 2, -1, 5], sim_step, sim_steps_per_env_step, env_id, assets_path)
        self.target_size_max = target_size_max
        self.target_size_min = target_size_min
        self.target_size_divisor = target_size_divisor
        self.target_size_success_threshold = target_size_success_threshold

        self.ground_plate: GroundPlate = None
        self.table: Box = None
        self.target_max_distance = 0.8
        self.target_size = target_size_max

    def set_up(self):
        self.ground_plate = GroundPlate(False)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -1]))  # move it lower because our floor is lower in this experiment

        # add the table the robot is standing on
        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5], seen_by_obstacle_sensor=False)
        self.table.build()

    def reset(self, success_rate: float):
        def generate_random_point_in_sphere(radius):
            while True:
                # Generate random points in the cube [-radius, radius]^3
                point = np.random.uniform(-radius, radius, size=3)

                # Check if the point is valid
                try:
                    if point[2] > 0.1 and np.linalg.norm(point) <= radius and np.linalg.norm(self.robots[0].position_rotation_sensor.position - point) > self.target_size + 0.1:
                        return point
                except TypeError:
                    # for some reason position is not initialized on first reset. Not beautiful but ok
                    if point[2] > 0.1 and np.linalg.norm(point) <= radius:
                        return point

        self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)

        # set target size
        if success_rate > self.target_size_success_threshold and self.target_size > self.target_size_min:
            self.target_size /= self.target_size_divisor

        # set target for goal
        self.position_targets = [generate_random_point_in_sphere(self.target_max_distance), self.target_size]

    def update(self):
        pass

