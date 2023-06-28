from typing import Tuple

import numpy as np
from gym.spaces import Box

from modular_drl_env.goal import Goal
from modular_drl_env.robot import Robot
from modular_drl_env.sensor import ObstacleSensor, ObstacleAbsoluteSensor

from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

goal_size = 0.1

dirac = 0.1
d_min = 0.2
k = 6
reward_collision=-1000
a = ((reward_collision) - (0)) / (np.exp(0.05) - 1)
b = (0) - a
reward_success=1000
lambda_1 = 1000.
lambda_2 = 500.
lambda_3 = 10.
lambda_4 = 0.

def f_dist_reward(x):
    return (0.5 * (x ** 2)) if x < dirac else (dirac * (x - 0.5 * dirac))

def f_obst_reward(x):
    return (d_min / (x + d_min)) ** k

def f_joint_limit_reward(x):
    return (a * np.exp(-(x - 0.05)) + b) if x <= 0.05 else 0


class PeterTestGoal(Goal):

    def get_observation_space_element(self) -> dict:
        ret = dict()
        ret["target"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["ee_position"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        return ret

    def get_observation(self) -> dict:
        self.position = self.robot.position_rotation_sensor.position
        self.distance = np.linalg.norm(self.goal - self.position)

        ret = dict()
        ret["target"] = self.goal
        ret["ee_position"] = self.position
        return ret

    """def reward(self, step, action) -> Tuple[float, bool, bool, bool, bool]:
        dist = abs(np.linalg.norm(self.position - self.goal))
        if pyb_u.collision:
            rew = -10000 - 2*self.last_reward
            done = True
            success = False
            timeout = False
        elif step > self.max_steps:
            rew = -5000 - 1.33*self.last_reward
            done = True
            success = False
            timeout = True
        elif dist <= goal_size:
            rew = self.max_steps/step * 100
            done = True
            success = True
            timeout = False
        else:
            #rew = -(dist**2)
            rew = (self.init_dist - dist)*10
            done = False
            success = False
            timeout = False

        self.last_reward = rew

        return rew, success, done, timeout, False"""

    def reward(self, step, action):
        reward = 0

        dist_reward = -f_dist_reward(self.distance)
        obst_reward = -f_obst_reward(self.obst_sensor.min_dist)
        action_reward = -np.sum(np.square(action))
        # penalty for being very close to joint limits
        dist_to_max = abs(self.robot.joints_limits_upper - self.robot.joints_sensor.joints_angles)
        dist_to_min = abs(self.robot.joints_sensor.joints_angles - self.robot.joints_limits_lower)
        dist_both = min(min(dist_to_max), min(dist_to_min))
        joint_limit_reward = f_joint_limit_reward(dist_both)

        self.is_success = False
        self.collided = pyb_u.collision
        self.done = False
        self.out_of_bounds = False
        self.timeout = False

        if self.collided:
            self.done = True
            reward += reward_collision
        elif self.distance < goal_size:
            self.done = True
            self.is_success = True
            reward += reward_success
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward += reward_collision / 2
        else:
            if self.normalize_rewards:
                reward = (lambda_1 * dist_reward + lambda_2 * obst_reward + lambda_3 * action_reward + lambda_4 * joint_limit_reward) / lambda_1
            else:
                reward = lambda_1 * dist_reward + lambda_2 * obst_reward + lambda_3 * action_reward + lambda_4 * joint_limit_reward

        self.reward_value = reward
        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds

    def on_env_reset(self, success_rate):
        def generate_random_point_in_sphere(radius):
            while True:
                # Generate random points in the cube [-radius, radius]^3
                point = np.random.uniform(-radius, radius, size=3)

                # Check if the point is within the sphere
                if np.linalg.norm(point) <= radius:
                    if point[2] > 0.1:
                        return point

        self.goal = generate_random_point_in_sphere(0.8)
        if not hasattr(self, "sphere"):
            self.sphere = pyb_u.create_sphere(position=np.zeros([3]), mass=0, radius=goal_size, color=[0, 1, 0, 0.65],
                                              collision=False)
        pyb_u.set_base_pos_and_ori(self.sphere, self.goal, np.ones([4]))

        self.robot.moveto_joints(self.robot.resting_pose_angles, False)

        self.init_dist = abs(np.linalg.norm(self.robot.position_rotation_sensor.position - self.goal))
        if self.init_dist < goal_size * 1.5:
            self.on_env_reset(success_rate)

        self.obst_sensor = None
        for sensor in self.robot.sensors:
            if type(sensor) == ObstacleSensor or type(sensor) == ObstacleAbsoluteSensor:
                self.obst_sensor = sensor
                break
        if self.obst_sensor is None:
            raise Exception("This goal type needs an obstacle sensor to be present for its robot!")

        return [("", 0., False, True)]
