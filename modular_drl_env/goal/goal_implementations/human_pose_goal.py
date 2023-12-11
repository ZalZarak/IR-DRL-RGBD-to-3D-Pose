import numpy as np
from gym.spaces import Box

import pybullet as p
from modular_drl_env.RGBDto3Dpose.src.simulator import is_joint_valid
from modular_drl_env.goal import Goal, PositionCollisionGoalNoShakingProximityV2
from modular_drl_env.robot import Robot

from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u


"""d_min = 0.08
k = 6
dirac = 0.1
reward_collision = -100
a = ((reward_collision) - (0)) / (np.exp(0.05) - 1)
b = (0) - a
reward_success = 150
lambda_1 = 1000
lambda_2 = 500
lambda_3 = 10
lambda_4 = 0."""


class HumanPoseGoal(Goal):
    def __init__(self,
                 robot: Robot,
                 normalize_rewards: bool,
                 normalize_observations: bool,
                 train: bool,
                 add_to_logging: bool,
                 max_steps: int,
                 continue_after_success: bool,
                 reward_success=100,
                 reward_collision=-100,
                 dist_threshold_start=0.3,
                 dist_threshold_end=0.01,
                 dist_threshold_increment_start=0.01,
                 dist_threshold_increment_end=0.001,
                 dist_threshold_overwrite: float = None,
                 dist_threshold_change: float = 0.8,
                 better_compatability_obs_space: bool = True,
                 done_on_oob: bool = True,
                 d_min: float= 0.2,
                 k: int=6,
                 dirac: float=0.1,
                 lambda_1: float=1000,
                 lambda_2: float=500,
                 lambda_3: float=10):

        super().__init__(robot, normalize_rewards, normalize_observations, train, False, add_to_logging, max_steps, continue_after_success)

        self.reward_success = reward_success
        self.reward_collision = reward_collision
        self.distance_threshold_start = dist_threshold_start
        self.distance_threshold_end = dist_threshold_end
        self.distance_threshold_increment_start = dist_threshold_increment_start
        self.distance_threshold_increment_end = dist_threshold_increment_end
        self.distance_threshold_change = dist_threshold_change
        self.d_min = d_min
        self.k = k
        self.dirac = dirac
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.distance_threshold = dist_threshold_start if self.train else dist_threshold_end
        if dist_threshold_overwrite:  # allows to set a different startpoint from the outside
            self.distance_threshold = dist_threshold_overwrite

    def get_observation_space_element(self) -> dict:
        ret = dict()
        ret["target"] = Box(low=-5, high=5, shape=(3,), dtype=np.float32)
        ret["ee_position"] = Box(low=-5, high=5, shape=(3,), dtype=np.float32)
        ret["joints"] = Box(low=-50, high=50, shape=(24, 3), dtype=np.float32)
        return ret

    def get_observation(self) -> dict:
        ret = dict()

        ret["target"] = self.target_position
        ret["ee_position"] = self.robot.position_rotation_sensor.position
        try:
            self.joints = self.robot.world.sim.joints[:25, :].copy()
        except: # TypeError:
            self.joints = np.zeros((24, 3))
        for i, joint in enumerate(self.joints):
            if not is_joint_valid(joint):
                self.joints[i] = (-50, -50, -50)
        ret["joints"] = self.joints

    def reward(self, step, action):
        goal_distance = np.linalg.norm(self.target_position - self.robot.position_rotation_sensor.position)
        dist_reward = self.f_dist_reward(goal_distance)

        robot_id = pyb_u.pybullet_object_ids[self.robot.object_id]
        min_dist = float('inf')
        for joint in self.joints:
            for robot_joint_index in range(p.getNumJoints(robot_id)):
                # Get the position of the current link
                link_state = p.getLinkState(robot_id, robot_joint_index)
                link_position = link_state[0]

                dist = np.linalg.norm(joint-link_position)

                if dist < min_dist:
                    min_dist = dist

        obst_reward = self.f_obst_reward(min_dist)
        action_reward = self.f_action_award(action)

        self.is_success = False
        self.collided = pyb_u.collision
        self.done = False
        self.out_of_bounds = False
        self.timeout = False

        if self.collided:
            self.done = True
            reward = self.reward_collision
        elif goal_distance < self.distance_threshold:
            self.done = True
            self.is_success = True
            reward = self.reward_success
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward = self.reward_collision/2
        else:
            reward = self.lambda_1 * dist_reward + self.lambda_2 * obst_reward + self.lambda_3 * action_reward

        self.reward_value = reward
        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds

    def on_env_reset(self, success_rate):
        def generate_random_point_on_circle(height):
            max_deviation = np.sqrt(0.8**2 - height**2)
            while True:
                x,y = np.random.uniform(-max_deviation, max_deviation, size=2)
                if np.linalg.norm((x,y,height)) <= 0.8:
                    return np.array((x, y, height))

        if self.train:
            # calculate increment
            ratio_start_end = (self.distance_threshold - self.distance_threshold_end) / (self.distance_threshold_start - self.distance_threshold_end)
            increment = (self.distance_threshold_increment_start - self.distance_threshold_increment_end) * ratio_start_end + self.distance_threshold_increment_end
            if success_rate > self.distance_threshold_change and self.distance_threshold > self.distance_threshold_end:
                self.distance_threshold -= increment
                print(f"Threshold = {self.distance_threshold}")
            elif success_rate < self.distance_threshold_change and self.distance_threshold < self.distance_threshold_start:
                #self.distance_threshold += increment / 25  # upwards movement should be slower # DISABLED
                pass
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end


        if not hasattr(self, "sphere"):
            self.sphere = pyb_u.create_sphere(position=np.zeros([3]), mass=0, radius=self.distance_threshold, color=[0, 1, 0, 0.65],
                                              collision=False)
            pyb_u.create_sphere(position=np.zeros([3]), mass=0, radius=0.2, color=[1, 1, 0, 1],
                                collision=False)
        self.target_position = generate_random_point_on_circle(0.5)
        pyb_u.set_base_pos_and_ori(self.sphere, self.target_position, np.ones([4]))

        self.joints = np.ones((25, 3))*(-50)

        return [("distance_threshold", self.distance_threshold, True, True)]

    def build_visual_aux(self):
        self.aux_object_ids.append(pyb_u.create_sphere(position=self.target_position, mass=0, radius=self.distance_threshold, color=[0, 1, 0, 0.65], collision=False))

    def f_dist_reward(self, x):
        # return (0.5 * (x ** 2)) if x < dirac else (dirac * (x - 0.5 * dirac))
        return -((0.5 * (x ** 2)) if x < self.dirac else (self.dirac * (abs(x) - 0.5 * self.dirac)))

    def f_obst_reward(self, x):
        # return 0
        return -((self.d_min / (x + self.d_min)) ** self.k)

    def f_action_award(self, x):
        return -(np.linalg.norm(x) ** 2)

    def f_joint_limit_reward(self, x):
        return 0
        # return (a * np.exp(-(x - 0.05)) + b) if x <= 0.05 else 0


"""class HumanPoseGoal2(PositionCollisionGoalNoShakingProximityV2):

    def get_observation(self) -> dict:"""