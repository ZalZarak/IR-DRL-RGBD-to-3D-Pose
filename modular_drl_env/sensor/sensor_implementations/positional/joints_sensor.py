from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.robot.robot import Robot
from time import process_time
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
        'JointsSensor'
    ]

class JointsSensor(Sensor):

    def __init__(self, 
                 robot: Robot,
                 sim_step: float, 
                 sim_steps_per_env_step: int,
                 add_joint_velocities: bool=False,
                 normalize: bool=False, 
                 add_to_observation_space:bool=True, 
                 add_to_logging: bool=True, 
                 update_steps: int=1
                 ):

        super().__init__(sim_step, sim_steps_per_env_step, normalize, add_to_observation_space, add_to_logging,  update_steps)
        
        # set associated robot
        self.robot = robot

        # set output data field name
        self.output_name = "joints_angles_" + self.robot.name
        self.output_name_vels = "joints_velocities_" + self.robot.name

        # init data storage
        self.joints_dims = len(self.robot.joints_limits_lower)
        self.joints_angles = None
        self.joints_velocities = None

        # whether to add joint velocities to observation space
        self.add_joint_velocities = add_joint_velocities

        # normalizing constants for faster normalizing
        self.normalizing_constant_a = 2 / self.robot.joints_range
        self.normalizing_constant_b = np.ones(self.joints_dims) - np.multiply(self.normalizing_constant_a, self.robot.joints_limits_upper)
        self.normalizing_constant_a_vels = 2 / (2 * self.robot.joints_max_velocities)
        self.normalizing_constant_b_vels = np.ones(self.joints_dims) - np.multiply(self.normalizing_constant_a_vels, self.robot.joints_max_velocities)

        #self.update()


    def update(self, step) -> dict:
        self.cpu_epoch = process_time()
        if step % self.update_steps == 0:
            self.joints_angles, self.joints_velocities = pyb_u.get_joint_states(self.robot.object_id, self.robot.controlled_joints_ids)
        self.cpu_time = process_time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = process_time()
        self.joints_angles, _ = pyb_u.get_joint_states(self.robot.object_id, self.robot.controlled_joints_ids)
        self.joints_velocities = np.zeros(self.joints_dims)
        self.cpu_time = process_time() - self.cpu_epoch

    def get_observation(self) -> dict:
        if self.normalize:
            return self._normalize()
        else:
            ret_dict = {self.output_name: self.joints_angles}
            if self.add_joint_velocities:
                ret_dict[self.output_name_vels] = self.joints_velocities
            return ret_dict

    def _normalize(self) -> dict:
        ret_dict = {self.output_name: np.multiply(self.normalizing_constant_a, self.joints_angles) + self.normalizing_constant_b}
        if self.add_joint_velocities:
            ret_dict[self.output_name_vels] = np.multiply(self.normalizing_constant_a_vels, self.joints_velocities) + self.normalizing_constant_b_vels
        return ret_dict

    def get_observation_space_element(self) -> dict:
        
        if self.add_to_observation_space:
            obs_sp_ele = dict()

            if self.normalize:
                obs_sp_ele[self.output_name] = Box(low=-1, high=1, shape=(self.joints_dims,), dtype=np.float32)
                if self.add_joint_velocities:
                    obs_sp_ele[self.output_name_vels] = Box(low=-1, high=1, shape=(self.joints_dims,), dtype=np.float32)
            else:
                obs_sp_ele[self.output_name] = Box(low=np.float32(self.robot.joints_limits_lower), high=np.float32(self.robot.joints_limits_upper), shape=(self.joints_dims,), dtype=np.float32)
                if self.add_joint_velocities:
                    obs_sp_ele[self.output_name_vels] = Box(low=-np.float32(self.robot.joints_max_velocities), high=np.float32(self.robot.joints_max_velocities), shape=(self.joints_dims,), dtype=np.float32)

            return obs_sp_ele
        else:
            return {}

    def get_data_for_logging(self) -> dict:
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict["joints_angles_" + self.robot.name] = self.joints_angles
        logging_dict["joints_velocities_" + self.robot.name] = self.joints_velocities
        logging_dict["joints_sensor_cpu_time_" + self.robot.name] = self.cpu_time

        return logging_dict

