from typing import Tuple

import numpy as np

from modular_drl_env.goal import Goal
from modular_drl_env.robot import Robot
from gym.spaces import Box
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u


class PeterGoal(Goal):
    def __init__(self, robot: Robot, normalize_rewards: bool, normalize_observations: bool, train: bool, add_to_observation_space: bool,
                 add_to_logging: bool, max_steps: int, continue_after_success: bool, target_size_eval: float,
                 reward_success=100., reward_collision=-100., reward_timeout=-50., delta_dif_target_size=0.1, d_ref=0.2, p=6, c1=1., c2=0.01, c3=0):
        super().__init__(robot, normalize_rewards, normalize_observations, train, add_to_observation_space, add_to_logging, max_steps, continue_after_success)

        self.target_size_eval = target_size_eval
        self.derive_delta_from_target_size = lambda: self.target_size + 0.1
        self.d_ref = 0.2
        self.p = 6
        self.c1 = 1
        self.c2 = 0.01
        self.c3 = 0
        self.reward_success = 100
        self.reward_collision = -100
        self.reward_timeout = -50

        self.target_size: float = None
        self.target: np.ndarray = None
        self.distance: float = None
        self.delta: float = None

        if self.normalize_rewards:
            raise NotImplemented

    def get_observation_space_element(self) -> dict:
        ret = dict()
        ret["target"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["ee_position"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["difference_goal"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["distance_goal"] = Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        return ret

    def get_observation(self) -> dict:
        ee_position = self.robot.position_rotation_sensor.position
        dif = self.target - ee_position
        self.distance = np.linalg.norm(dif)

        ret = dict()
        ret["target"] = self.target
        ret["ee_position"] = ee_position
        ret["difference_goal"] = dif
        ret["distance_goal"] = np.array([self.distance])
        return ret

    def reward(self, step, action) -> Tuple[float, bool, bool, bool, bool]:
        def f_dist_reward(x):
            return -((0.5 * (x ** 2)) if x < self.delta else (self.delta * (x - 0.5 * self.delta)))

        def f_action_award(x):
            return -(np.linalg.norm(x) ** 2)

        def f_obst_reward(x):
            return -((self.d_ref / (x + self.d_ref)) ** self.p)

        reward = 0

        dist_reward = f_dist_reward(self.distance)
        action_reward = f_action_award(action)
        obst_reward = 0  # f_obst_reward(self.obst_sensor.min_dist)

        # penalty for being very close to joint limits
        """dist_to_max = abs(self.robot.joints_limits_upper - self.robot.joints_sensor.joints_angles)
        dist_to_min = abs(self.robot.joints_sensor.joints_angles - self.robot.joints_limits_lower)
        dist_both = min(min(dist_to_max), min(dist_to_min))
        joint_limit_reward = f_joint_limit_reward(dist_both)"""

        success = False
        collided = pyb_u.collision
        done = False
        out_of_bounds = False
        timeout = False

        if collided:
            done = True
            reward += self.reward_collision
        elif self.distance < self.target_size:
            done = True
            success = True
            reward += self.reward_success
        elif step > self.max_steps:
            done = True
            timeout = True
            reward += self.reward_timeout
        else:
            """if self.normalize_rewards:
                raise NotImplemented"""
            reward = self.c1 * dist_reward + self.c2 * action_reward + self.c3 * obst_reward

        return reward, success, done, timeout, out_of_bounds

    def on_env_reset(self, success_rate):
        self.target, self.target_size = self.robot.world.position_targets
        if not self.train:
            self.target_size = self.target_size_eval
        self.delta = self.derive_delta_from_target_size()

        return [("target_size", self.target_size, True, True)]

    def build_visual_aux(self):
        self.aux_object_ids.append(pyb_u.create_sphere(position=self.target, mass=0, radius=self.target_size, color=[0, 1, 0, 0.65], collision=False))
