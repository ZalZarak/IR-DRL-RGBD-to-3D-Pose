from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

class Obstacle(ABC):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, sim_step: float, sim_steps_per_env_step: int, velocity: float=0) -> None:

        # current and initial position
        self.position = np.array(position)
        self.orientation = np.array(rotation)
        self.position_orig = np.array(position)
        self.orientation_orig = np.array(rotation)

        # velocity vector, when moving
        self.velocity = np.zeros(3)

        # pybullet object id, gets set through build method
        self.object_id = None

        # (potential) trajectory, measured relative to position orig
        # if this has no element, the obstacle will not move
        # if this has one element, the obstalce will move towards it and stay there
        # for two or more elements the obstacle will loop between the two or more points
        self.trajectory = [np.array(ele) for ele in trajectory]
        self.sim_step = sim_step  # amount of time that passes per physics step
        self.sim_steps_per_env_step = sim_steps_per_env_step
        self.move_step = velocity * (sim_step * sim_steps_per_env_step)  # distance that is travelled per env step
        self.trajectory_idx = -1
        self.closeness_threshold = 1e-3  # to determine if two positions are the same

    @abstractmethod
    def build(self) -> int:
        """
        This method should spawn the obstalce into the simulation.
        Always use the inital position.
        Must return the object ID of the obstacle.
        """
        return 0

    def move_traj(self):
        """
        Moves the obstacle along the trajectory with constant velocity.
        """
        if not self.trajectory:
            pass  # empty trajectory, do nothing
        elif len(self.trajectory) == 1:
            # move towards the one goal
            goal = self.trajectory[0] + self.position_orig
            diff = goal - self.position
            diff_norm = np.linalg.norm(diff)
            if diff_norm <= self.closeness_threshold:
                self.trajectory.pop(0)  # next time the move method will do nothing
            else:
                move_step = self.move_step if diff_norm > self.move_step else diff_norm # ensures that we don't jump over the target destination
                step = diff * (move_step / diff_norm)
                self.velocity = step / (self.sim_step * self.sim_steps_per_env_step)
                self.position = self.position + step
                pyb_u.set_base_pos_and_ori(object_id=self.object_id, position=self.position, orientation=self.orientation)
        else:  # looping trajectory
            goal = self.trajectory[self.trajectory_idx + 1] + self.position_orig
            diff = goal - self.position
            diff_norm = np.linalg.norm(diff)
            if diff_norm <= self.closeness_threshold:
                self.trajectory_idx += 1
                # loop back again
                if self.trajectory_idx > len(self.trajectory) - 2:
                    self.trajectory_idx = -1
            else:
                move_step = self.move_step if diff_norm > self.move_step else diff_norm # ensures that we don't jump over the target destination
                step = diff * (move_step / diff_norm) 
                self.velocity = step / (self.sim_step * self.sim_steps_per_env_step) 
                self.position = self.position + step
                pyb_u.set_base_pos_and_ori(object_id=self.object_id, position=self.position, orientation=self.orientation)

    def move_base(self, new_base_position, new_base_rotation=None):
        if new_base_rotation is None:
            new_base_rotation = self.orientation_orig
        pyb_u.set_base_pos_and_ori(object_id=self.object_id, position=new_base_position, orientation=new_base_rotation)
        self.position_orig = new_base_position
        self.position = new_base_position
        self.orientation = new_base_rotation
        self.orientation_orig = new_base_rotation
        self.velocity = np.zeros(3)
