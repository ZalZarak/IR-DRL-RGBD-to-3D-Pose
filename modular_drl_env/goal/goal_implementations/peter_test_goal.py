from typing import Tuple

import numpy as np
from gym.spaces import Box

from modular_drl_env.goal import Goal
from modular_drl_env.robot import Robot
from modular_drl_env.sensor import ObstacleSensor, ObstacleAbsoluteSensor

from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

d_min = 0.2
k = 6
reward_collision = -100
# a = ((reward_collision) - (0)) / (np.exp(0.05) - 1)
# b = (0) - a
reward_success = 100
lambda_1 = 1
lambda_2 = 0#60
lambda_3 = 0 #0.01
lambda_4 = 0.

def f_dist_reward(x, dirac):
    # return (0.5 * (x ** 2)) if x < dirac else (dirac * (x - 0.5 * dirac))
    return -((0.5 * (x ** 2)) if x < dirac else (dirac * (abs(x) - 0.5 * dirac)))

def f_obst_reward(x):
    return 0
    #return -((d_min / (x + d_min)) ** k)

def f_action_award(x):
    return -(np.linalg.norm(x)**2)

def f_joint_limit_reward(x):
    return 0
    # return (a * np.exp(-(x - 0.05)) + b) if x <= 0.05 else 0


class PeterTestGoal(Goal):
    goal_size = 0.1
    dirac = 0.6


    def get_observation_space_element(self) -> dict:
        ret = dict()
        ret["target"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["ee_position"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["difference_goal"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["distance_goal"] = Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        #ret["joints"] = Box(low=-50, high=50, shape=(24, 3), dtype=np.float32)
        return ret

    def get_observation(self) -> dict:
        self.position = self.robot.position_rotation_sensor.position
        dif = self.goal - self.position
        self.distance = np.linalg.norm(dif)

        ret = dict()
        ret["target"] = self.goal
        ret["ee_position"] = self.position
        ret["difference_goal"] = dif
        ret["distance_goal"] = np.array([self.distance])
        """try:
            ret["joints"] = self.robot.world.sim.joints[:25, :].copy()
        except AttributeError:
            ret["joints"] = np.zeros([24, 3])"""
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

        dist_reward = f_dist_reward(self.distance, self.dirac)
        obst_reward = 0 #f_obst_reward(self.obst_sensor.min_dist)
        action_reward = f_action_award(action)
        # penalty for being very close to joint limits
        """dist_to_max = abs(self.robot.joints_limits_upper - self.robot.joints_sensor.joints_angles)
        dist_to_min = abs(self.robot.joints_sensor.joints_angles - self.robot.joints_limits_lower)
        dist_both = min(min(dist_to_max), min(dist_to_min))
        joint_limit_reward = f_joint_limit_reward(dist_both)"""

        self.is_success = False
        self.collided = pyb_u.collision
        self.done = False
        self.out_of_bounds = False
        self.timeout = False

        if self.collided:
            self.done = True
            reward += reward_collision
        elif self.distance < self.goal_size:
            self.done = True
            self.is_success = True
            reward += reward_success
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward += reward_collision / 2
        else:
            """if self.normalize_rewards:
                reward = (lambda_1 * dist_reward + lambda_2 * obst_reward + lambda_3 * action_reward) / lambda_1
            else:
                reward = lambda_1 * dist_reward + lambda_2 * obst_reward + lambda_3 * action_reward
                print(f"Reward: {reward}")"""
            reward = lambda_1 * dist_reward + lambda_2 * obst_reward + lambda_3 * action_reward
            """print(f"DistReward: {lambda_1 * dist_reward}")
            print(f"ObstReward: {lambda_2 * obst_reward}")
            print(f"ActiReward: {lambda_3 * action_reward}")
            print(f"Reward: {reward}")"""

        self.reward_value = reward
        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds

    def on_env_reset(self, success_rate):

        def update_dirac(self):
            # FIXME self.dirac?
            self.delta = self.goal_size + 0.1

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
            self.sphere = pyb_u.create_sphere(position=np.zeros([3]), mass=0, radius=self.goal_size, color=[0, 1, 0, 0.65],
                                              collision=False)
        pyb_u.set_base_pos_and_ori(self.sphere, self.goal, np.ones([4]))

        if success_rate > 0.9 and self.goal_size > 0.1:
            self.goal_size /= 1.2
            pyb_u.remove_object(self.sphere)
            self.sphere = pyb_u.create_sphere(position=np.zeros([3]), mass=0, radius=self.goal_size,
                                              color=[0, 1, 0, 0.65], collision=False)
            update_dirac(self)
            print(f"Goal size is {self.goal_size} and dirac is {self.dirac}")


        self.robot.moveto_joints(self.robot.resting_pose_angles, False)

        self.init_dist = abs(np.linalg.norm(self.robot.position_rotation_sensor.position - self.goal))
        if self.init_dist < self.goal_size * 1.5:
            self.on_env_reset(success_rate)

        """if not hasattr(self, "obst_sensor"):
            self.obst_sensor = None
            for sensor in self.robot.sensors:
                if type(sensor) == ObstacleSensor or type(sensor) == ObstacleAbsoluteSensor:
                    self.obst_sensor = sensor
                    break
            if self.obst_sensor is None:
                raise Exception("This goal type needs an obstacle sensor to be present for its robot!")"""

        # print(pyb_u.get_joint_states(self.robot.object_id, self.robot.controlled_joints_ids))

        return [("", 0., False, True)]
