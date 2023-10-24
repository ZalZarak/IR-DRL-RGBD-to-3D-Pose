import random
import sys
sys.path.append("modular_drl_env/RGBDto3Dpose")
import time

import numpy as np

from modular_drl_env.world import World
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.shapes import Box
from modular_drl_env.RGBDto3Dpose.src.simulator import Simulator
from modular_drl_env.RGBDto3Dpose.src.config import config
#from modular_drl_env.RGBDto3Dpose.simulator_adjusted import SimulatorAdjusted
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

class PeterTestWorld(World):
    def set_up(self):
        self.ground_plate = GroundPlate(False)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -1]))

        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5], seen_by_obstacle_sensor=False)
        self.table.build()

        self.sim = Simulator(**config["Simulator"])
        self.t = time.time()

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



    def reset(self, success_rate: float):
        # self.robot.reset()
        if self.sim.playback:   # random start
            playback_duration = self.sim.frames[-1][0]
            self.t = time.time() - random.uniform(0, playback_duration)

    def update(self):
        if self.sim.playback:
            self.sim.process_frame_at_time(time.time() - self.t)
        else:
            self.sim.process_frame_sync()