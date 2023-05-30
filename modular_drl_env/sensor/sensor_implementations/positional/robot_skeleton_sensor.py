from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.robot.robot import Robot
from time import process_time
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from typing import List, Tuple

__all__ = [
    "RobotSkeletonSensor"
]

def interpolate_3d(vec1, vec2, n_points):
    ret = []
    m = vec2 - vec1
    step = m / (n_points + 1)
    for i in range(1, n_points + 1):
        ret.append(vec1 + step * i)
    return np.array(ret)

class RobotSkeletonSensor(Sensor):
    """
    Sensor that reports the cartesian world position of each desired link of a robot.
    Adapted from the code written by Amir.
    """

    def __init__(self, normalize: bool, 
                 add_to_observation_space: bool, 
                 add_to_logging: bool, 
                 sim_step: float, 
                 update_steps: int, 
                 sim_steps_per_env_step: int,
                 robot: Robot,
                 reference_link_ids: List[str],
                 extra_points_link_pairs: List=[],
                 report_velocities: bool=False):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, sim_steps_per_env_step)

        # set associated robot
        self.robot = robot

        # set output data field names
        self.output_name = "pos_" + self.robot.name
        self.output_name_vels = "vel_" + self.robot.name

        # the links for which the sensor will work
        self.reference_link_ids = reference_link_ids

        # bool for also reporting velocities
        self.report_velocities = report_velocities

        # extra points via linear interpolation between selected links
        self.extra_points_link_pairs = extra_points_link_pairs
        if self.extra_points_link_pairs:
            print("[WARNING] You've activated extra points for the robot skeleton sensor of " + self.robot.name + ". Check via Pybullet GUI if their placement (red bubbles) makes sense as they're automated via simple linear intepolation!")
        self.num_extra = sum([tup[2] for tup in self.extra_points_link_pairs])
        self.extra_points_coordinates = np.zeros((self.num_extra, 3))

        # normalizing constants for faster normalizing
        self.normalizing_constant_a_pos = 2 / (np.ones(3) * 100)  # this is equivalent to a maximum workspace of 50 m in each direction
        self.normalizing_constant_b_pos = np.ones(3) - np.multiply(self.normalizing_constant_a_pos, np.ones(3) * 50)
        self.normalizing_constant_a_vel = 2 / (np.ones(3) * 50)  # arbitrary max of 25 m/s, we don't really know the max speed of robot end effectors in general
        self.normalizing_constant_b_vel = np.ones(3) - np.multiply(self.normalizing_constant_b_pos, np.ones(3) * 25)

        # data storage
        self.positions = {}
        self.velocities = {}
        for link in self.reference_link_ids:
            self.positions[link] = np.zeros(3)
            self.velocities[link] = np.zeros(3)

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()

            if self.normalize:
                for link in self.reference_link_ids:
                    ret[self.output_name + "_" + link] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                    if self.report_velocities:
                        ret[self.output_name_vels + "_" + link] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                if self.extra_points_link_pairs:
                    for idx, _ in enumerate(self.extra_points_coordinates):
                        ret[self.output_name + "_extra_point" + str(idx)] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            else:
                for link in self.reference_link_ids:
                    ret[self.output_name + "_" + link] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
                    if self.report_velocities:
                        ret[self.output_name_vels + "_" + link] = Box(low=-25, high=25, shape=(3,), dtype=np.float32)
                if self.extra_points_link_pairs:
                    for idx, _ in enumerate(self.extra_points_coordinates):
                        ret[self.output_name + "_extra_point" + str(idx)] = Box(low=-50, high=50, shape=(3,), dtype=np.float32)
            return ret
        else:
            return {}
        
    def get_observation(self) -> dict:
        ret = dict()
        if self.normalize:
            for link in self.reference_link_ids:
                ret[self.output_name + "_" + link] = np.multiply(self.normalizing_constant_a_pos, self.positions[link]) + self.normalizing_constant_b_pos
                if self.report_velocities:
                    ret[self.output_name_vels + "_" + link] = np.multiply(self.normalizing_constant_a_vel, self.velocities[link]) + self.normalizing_constant_b_vel
            if self.extra_points_link_pairs:
                for idx, point in enumerate(self.extra_points_coordinates):
                    ret[self.output_name + "_extra_point" + str(idx)] = np.multiply(self.normalizing_constant_a_pos, point) + self.normalizing_constant_b_pos
        else:
            for link in self.reference_link_ids:
                ret[self.output_name + "_" + link] = self.positions[link]
                if self.report_velocities:
                    ret[self.output_name_vels + "_" + link] = self.velocities[link]
            if self.extra_points_link_pairs:
                for idx, point in enumerate(self.extra_points_coordinates):
                    ret[self.output_name + "_extra_point" + str(idx)] = point
        return ret
    
    def update(self, step) -> dict:
        self.cpu_epoch = process_time()
        if step % self.update_steps == 0:
            pos, _, vel, _ = pyb_u.get_link_states(self.robot.object_id, self.reference_link_ids)
            for idx, link in enumerate(self.reference_link_ids):
                self.positions[link] = pos[idx]
                self.velocities[link] = vel[idx]
            idx = 0
            for tup in self.extra_points_link_pairs:
                link1, link2, num = tup
                self.extra_points_coordinates[idx:idx + num] = interpolate_3d(self.positions[link1], self.positions[link2], num)
                idx += num       
        self.cpu_time = process_time() - self.cpu_epoch

        return self.get_observation()
    
    def _normalize(self) -> dict:
        pass

    def reset(self):
        pos, _, vel, _ = pyb_u.get_link_states(self.robot.object_id, self.reference_link_ids)
        for idx, link in enumerate(self.reference_link_ids):
            self.positions[link] = pos[idx]
            self.velocities[link] = vel[idx]
            idx = 0
            for tup in self.extra_points_link_pairs:
                link1, link2, num = tup
                self.extra_points_coordinates[idx:idx + num] = interpolate_3d(self.positions[link1], self.positions[link2], num)
                idx += num

    def get_data_for_logging(self) -> dict:
        # TODO
        return {}
    
    def build_visual_aux(self):
        for link in self.reference_link_ids:
            velocity = np.linalg.norm(self.velocities[link])
            radius = 0.015 * velocity + 0.02
            vis_sphere = pyb_u.create_sphere(self.positions[link], 0, radius, color = [0.294, 0, 0.51, 0.35], collision=False)
            self.aux_visual_objects.append(vis_sphere)
        if self.extra_points_link_pairs:
            for point in self.extra_points_coordinates:
                radius = 0.06
                vis_sphere = pyb_u.create_sphere(point, 0, radius, color = [1, 0, 0, 0.35], collision=False)
                self.aux_visual_objects.append(vis_sphere)
    
