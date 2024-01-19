import numpy as np
from gym.spaces import Box

from modular_drl_env.goal import PositionCollisionGoalNoShakingProximityV2, Goal
from modular_drl_env.robot import Robot
from modular_drl_env.sensor import ObstacleSensor
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u


d_min = 0.2
dirac = 0.1
k = 6
reward_success = 100
reward_collision = -100
lambda_1 = 1000
lambda_2 = 500
lambda_3 = 10
lambda_4 = 0
goal_size_max = 0.5
goal_size_min = 0.05
goal_size_change_factor = 1.1
goal_size_eval = 0.1
goal_size_change_success_rate = 0.8

def f_dist_reward(x):
    # return (0.5 * (x ** 2)) if x < dirac else (dirac * (x - 0.5 * dirac))
    return -((0.5 * (x ** 2)) if x < dirac else (dirac * (abs(x) - 0.5 * dirac)))

def f_obst_reward(x):
    return -((d_min / (x + d_min)) ** k)

def f_action_award(x):
    return -(np.linalg.norm(x)**2)


a = (reward_collision - 0) / (np.exp(0.05) - 1)
b = 0 - a
def f_joint_limit_reward(x):
    return (a * np.exp(-(x - 0.05)) + b) if x <= 0.05 else 0


class HumanPoseGoal(Goal):
    """
    Goal to work with HumanPoseWorld. Ignores normalization.
    """

    def __init__(self, robot: Robot, normalize_rewards: bool, normalize_observations: bool, train: bool, add_to_observation_space: bool,
                 add_to_logging: bool, max_steps: int):

        super().__init__(robot, normalize_rewards, normalize_observations, train, add_to_observation_space, add_to_logging, max_steps)

        self.goal_size = goal_size_max if self.train else goal_size_eval
        self.sphere = pyb_u.create_sphere(position=-np.array([0,0,0]), mass=0, radius=self.goal_size, color=[0, 1, 0, 0.65],
                                          collision=False)
        self.metric_names = ["goal_size"]

        if not hasattr(self, "obst_sensor"):
            self.obst_sensor = None
            for sensor in self.robot.sensors:
                if type(sensor) == ObstacleSensor:
                    self.obst_sensor = sensor
                    break
            if self.obst_sensor is None:
                raise Exception("This goal type needs an obstacle sensor to be present for its robot!")

    def get_observation_space_element(self) -> dict:
        ret = dict()
        ret["target"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["ee_position"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["difference_goal"] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
        ret["distance_goal"] = Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        # ret["joints"] = Box(low=-50, high=50, shape=(24, 3), dtype=np.float32)
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
        except AttributeError and TypeError:
            ret["joints"] = np.zeros([24, 3])"""
        return ret

    def reward(self, step, action):
        reward = 0

        dist_reward = f_dist_reward(self.distance)
        obst_reward = f_obst_reward(self.obst_sensor.min_dist)
        action_reward = f_action_award(action)

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
            reward = lambda_1 * dist_reward + lambda_2 * obst_reward + lambda_3 * action_reward
            """print(f"DistReward: {lambda_1 * dist_reward}")
            print(f"ObstReward: {lambda_2 * obst_reward}")
            print(f"ActiReward: {lambda_3 * action_reward}")
            print(f"Reward: {reward}")"""

        self.reward_value = reward
        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds

    def on_env_reset(self, success_rate):
        def generate_random_point_in_sphere(radius):
            while True:
                # Generate random points in the cube [-radius, radius]^3
                x,y = np.random.uniform(-radius, radius, size=2)
                z = np.random.uniform(0.1, radius, size=1)[0]
                point = np.array([x, y, z])

                # Check if the point is within the sphere and goal would be a bit away from robot
                if np.linalg.norm(point) <= radius and np.linalg.norm(self.robot.position_rotation_sensor.position - point) > self.goal_size * 1.5:
                    return point

        self.robot.moveto_joints(self.robot.resting_pose_angles, False)

        if success_rate > goal_size_change_success_rate and self.goal_size > goal_size_min and self.train:
            self.goal_size /= goal_size_change_factor
            pyb_u.remove_object(self.sphere)
            self.sphere = pyb_u.create_sphere(position=np.zeros([3]), mass=0, radius=self.goal_size,
                                              color=[0, 1, 0, 0.65], collision=False)
            print(f"Goal size is {self.goal_size}")

        self.goal = generate_random_point_in_sphere(0.8)
        pyb_u.set_base_pos_and_ori(self.sphere, self.goal, np.ones([4]))

        return [("goal_size", self.goal_size, False, True)]


class HumanPoseGoal2(PositionCollisionGoalNoShakingProximityV2):
    def get_observation(self) -> dict:
        # get the data
        self.position = self.robot.position_rotation_sensor.position
        self.target = self.robot.world.target
        dif = self.target - self.position
        self.distance = np.linalg.norm(dif)

        self.past_distances.append(self.distance)
        if len(self.past_distances) > 10:
            self.past_distances.pop(0)

        ret = np.zeros(4)
        ret[:3] = dif
        ret[3] = self.distance

        if self.normalize_observations:
            return {self.output_name: np.multiply(self.normalizing_constant_a_obs, ret) + self.normalizing_constant_b_obs}
        else:
            return {self.output_name: ret}

    def build_visual_aux(self):
        # build a sphere of distance_threshold size around the target
        self.target = self.robot.world.target  # self.robot.world.position_targets[self.robot.mgt_id]
        self.aux_object_ids.append(pyb_u.create_sphere(position=self.target, mass=0, radius=self.distance_threshold, color=[0, 1, 0, 0.65], collision=False))
