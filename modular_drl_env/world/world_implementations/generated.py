from modular_drl_env.world.world import World
from modular_drl_env.world.obstacles.shapes import Box, Sphere, Cylinder
from modular_drl_env.world.obstacles.human import Human
import numpy as np
import yaml
import os
import math
import glob
from random import choice, shuffle
#from modular_drl_env.shared.maze_generator import MazeGenerator
from modular_drl_env.shared.shelf_generator import ShelfGenerator
from modular_drl_env.world.obstacles.obstacle import Obstacle
#from modular_drl_env.world.obstacles.maze.maze import MazeObstacle
from modular_drl_env.world.obstacles.shelf.shelf import ShelfObstacle
from modular_drl_env.world.obstacles.urdf_object import URDFObject

__all__ = [
    'GeneratedWorld'
]

URDF_PATH = "./assets/"
def findUrdfs(search_name):
    return list(glob.iglob(os.path.join(URDF_PATH, f"**/{search_name}.urdf"), recursive=True))

def getTrajectory(obj):
    if "params" not in obj or "move" not in obj["params"]:
        return []
    return list(map(lambda x: np.array(x), obj["params"]["move"]))

def getVel(obj):
    if "params" not in obj or "vel" not in obj["params"]:
        return .1
    return obj["params"]["vel"]

def getScale(obj):
    if "scale" in obj:
        scale = obj["scale"]
    else:
        scale = 1
    return scale

class GeneratedWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed between the p
    Depending on the configuration, some of these can be moving in various directions at various speeds
    """

    def __init__(self, workspace_boundaries: list,
                       sim_step: float,
                       sim_steps_per_env_step: int,
                       env_id: int,
                       assets_path: str,
                       obstacles: dict,
                       start_override: dict={},
                       ):
        """
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param sim_step: float for the time per sim step
        """
        super().__init__(workspace_boundaries, sim_step, sim_steps_per_env_step, env_id, assets_path)
        self.config = obstacles 
        self.start_override = start_override


    def load_obstacle(self, obstacle):
        obstacle_name = obstacle["type"]
        position = obstacle["position"]
        rotation = obstacle["rotation"]
        scale = getScale(obstacle)
        vel = getVel(obstacle)
        trajectory = getTrajectory(obstacle)

        if obstacle_name == "human":
            self.obstacle_objects.append(Human(position, rotation, trajectory, self.sim_step, 1, scale))
        # elif obstacle_name == "maze":
        #     generator = MazeGenerator(obstacle["params"])
        #     urdf_name = self.assets_path + "/runtime/maze_" + str(self.env_id) + ".urdf"
        #     generator.generate_to_file(urdf_name)
        #     self.obstacle_objects.append(URDFObject(position, rotation, trajectory, self.sim_step, self.sim_steps_per_env_step, vel, urdf_name, scale))
        elif obstacle_name == "shelf":
            generator = ShelfGenerator(obstacle["params"])
            urdf_name = self.assets_path + "/runtime/shelf_" + str(self.env_id) + ".urdf"
            generator.generate_to_file(urdf_name)
            self.obstacle_objects.append(URDFObject(position, rotation, trajectory, self.sim_step, self.sim_steps_per_env_step, vel, urdf_name, scale))
        elif obstacle_name == "box":
            self.obstacle_objects.append(Box(position, rotation, trajectory, self.sim_step, self.sim_steps_per_env_step, vel, **obstacle["params"]))
        elif obstacle_name == "sphere":
            self.obstacle_objects.append(Sphere(position, rotation, trajectory, self.sim_step, self.sim_steps_per_env_step, vel, **obstacle["params"]))
        elif obstacle_name == "cylinder":
            self.obstacle_objects.append(Cylinder(position, rotation, trajectory, self.sim_step, self.sim_steps_per_env_step, vel, **obstacle["params"]))
        elif obstacle_name == "basic":
            urdfs = findUrdfs(obstacle["urdf"])
            if len(urdfs) > 0:
                urdf_name = urdfs[0]
            else:
                urdf_name = f"{urdf_name}.urdf"
            self.obstacle_objects.append(URDFObject(position, rotation, trajectory, self.sim_step, self.sim_steps_per_env_step, vel, urdf_name, scale))

    def set_up(self):
        # load obstacle configs
        for obstacle in self.config:
            self.load_obstacle(obstacle)

        # build obstacles into world
        for obstacle in self.obstacle_objects:
            obstacle.build()

    def reset(self, success_rate: float):
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        # move back any moved objects to starting position
        for obst in self.obstacle_objects:
            obst.move_base(obst.position_orig)

        # generate targets and starting positions
        robots_with_starting_points = [robot for robot in self.robots if robot.goal is not None]
        if self.start_override:
            for robot in robots_with_starting_points:
                robot.moveto_joints(robot.resting_pose_angles, False)
                robot.position_rotation_sensor.reset()
                self.ee_starting_points.append((robot.position_rotation_sensor.position, robot.position_rotation_sensor.rotation, robot.resting_pose_angles))
        else:
            self._create_ee_starting_points(robots_with_starting_points, factor=success_rate**3)
        self._create_position_and_rotation_targets(robots_with_starting_points, min_dist=0.01)

        # move robots to starting position
        for idx, robot in enumerate(self.robots):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)

    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.move_traj()