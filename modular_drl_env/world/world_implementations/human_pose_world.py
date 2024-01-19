import random
import sys
sys.path.append("modular_drl_env/RGBDto3Dpose")
import time

import numpy as np
import pybullet as p

from modular_drl_env.world import World
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.shapes import Box, Sphere, Cylinder
from modular_drl_env.RGBDto3Dpose.src.simulator import Simulator
from modular_drl_env.RGBDto3Dpose.src.config import config
#from modular_drl_env.RGBDto3Dpose.simulator_adjusted import SimulatorAdjusted
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

class HumanPoseWorld(World):
    """
    A world which works with RGDBto3DPose. Creates Human Geometry (which should be added automatically to ObstacleSensor and Observation...).
    For HumanPoseGoal and HumanPoseGoal2.
    """

    def set_up(self):
        self.ground_plate = GroundPlate(False)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -0.75]))

        self.table = Box(np.array([0.35, 0.583, -0.375]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0,
                         [0.415, 0.648, 0.375], seen_by_obstacle_sensor=True)
        self.table.build()

        c = config["Simulator"]
        c["build_geometry"] = False
        self.sim = Simulator(**c)

        # pregenerate geometry and add to sim
        for limb in self.sim.limbs:
            if len(limb) == 1:  # if this limb has only one point, like head or hand, it's simulated with a sphere
                n = Sphere(position=[0, 0, 0], trajectory=[], sim_step=self.sim_step, sim_steps_per_env_step=self.sim_steps_per_env_step,
                           radius=self.sim.radii[limb[0]], color=[0, 0, 0, 0], seen_by_obstacle_sensor=True)
                n = n.build()
                self.sim.limbs_pb[(self.sim.joint_map[limb[0]],)] = pyb_u.pybullet_object_ids[n]
            elif len(limb) == 2:  # if this limb has two points, like head or hand, it's simulated with a cylinder
                limb_str = f"{limb[0]}-{limb[1]}"
                if self.sim.joint_map[limb[0]] >= self.sim.joint_map[limb[1]]:
                    raise ValueError(f"limbs: for all tuples (a,b): joint_map[a] < joint_map[b], but joint_map[{limb[0]}] >= joint_map[{limb[1]}]")
                n = Cylinder(position=[0, 0, 0], rotation=[0, 0, 0, 1], trajectory=[], sim_step=self.sim_step,
                             sim_steps_per_env_step=self.sim_steps_per_env_step,
                             radius=self.sim.radii[limb_str], height=self.sim.lengths[limb_str], color=[0, 0, 0, 0], seen_by_obstacle_sensor=True)
                n = n.build()
                self.sim.limbs_pb[(self.sim.joint_map[limb[0]], self.sim.joint_map[limb[1]])] = pyb_u.pybullet_object_ids[n]
            else:
                raise ValueError(f"limbs: no connections between more then two joints: {limb}")

        # set collision filter so that limbs don't collide
        for body1 in self.sim.limbs_pb.values():
            p.setCollisionFilterGroupMask(body1, -1, 1, 0)

    def reset(self, success_rate: float):
        def generate_random_point_on_circle(height):
            max_deviation = np.sqrt(0.8**2 - height**2)
            while True:
                x,y = np.random.uniform(-max_deviation, max_deviation, size=2)
                if np.linalg.norm((x,y,height)) <= 0.8:
                    return np.array((x, y, height))

        def generate_random_point_in_sphere(radius):
            while True:
                # Generate random points in the cube [-radius, radius]^3
                point = np.random.uniform(-radius, radius, size=3)

                # Check if the point is valid
                if point[2] > 0.1 and np.linalg.norm(point) <= radius:
                    return point

        self.target = generate_random_point_in_sphere(0.8)  # only for HumanPoseGoal2, HumanPoseGoal makes its own goal

        self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)

        self.t = 0
        self.start_time = time.time()
        if self.sim.playback:   # random start
            playback_duration = self.sim.frames[-1][0]
            self.t = random.uniform(0, playback_duration)


    playback_speed_factor = 10   # since calculation of sim time is apperently wrong...
    def update(self):
        self.t += self.sim_step * self.sim_steps_per_env_step   # FIXME: too slow without factor
        if self.sim.playback:
            self.sim.process_frame_at_time(self.t * self.playback_speed_factor)
            #self.sim.process_frame_at_time((time.time() - self.start_time))
        else:
            self.sim.process_frame_sync()


class HumanPoseWorld2(HumanPoseWorld):
    """
    A world which works with RGDBto3DPose. Doesn't create human geometry. No Goal implemented for it, just to show how it would be done.
    """

    def set_up(self):
        self.ground_plate = GroundPlate(False)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -0.75]))

        self.table = Box(np.array([0.35, 0.583, -0.375]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.415, 0.648, 0.375], seen_by_obstacle_sensor=True)
        self.table.build()

        c = config["Simulator"]
        c["build_geometry"] = True
        self.sim = Simulator(**c)

        # add to pyb_u since objects were created outside of it.
        joint_map_rev = {v: k for k, v in config["Simulator"]["joint_map"].items()}
        for op_ids, p_id in self.sim.limbs_pb.items():
            if len(op_ids) == 1:
                pyb_u.pybullet_object_ids[joint_map_rev[op_ids[0]]] = p_id
                pyb_u.gym_env_str_names[p_id] = joint_map_rev[op_ids[0]]
            else:
                op_id_min, op_id_max = sorted(op_ids)
                name = f"{joint_map_rev[op_id_min]}-{joint_map_rev[op_id_max]}"
                pyb_u.pybullet_object_ids[name] = p_id
                pyb_u.gym_env_str_names[p_id] = name
