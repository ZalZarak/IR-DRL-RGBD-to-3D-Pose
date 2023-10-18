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

        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5], seen_by_obstacle_sensor=True)
        self.table.build()

        self.sim = Simulator(**config["Simulator"])
        # self.sim = SimulatorAdjusted(**config["Simulator"], sim_step=self.sim_step, sim_step_per_env_step=self.sim_steps_per_env_step)
        self.t = time.time()
        self.t_update = self.t



    def reset(self, success_rate: float):
        #self.robot.reset()
        pass

    def update(self):
        pass
        self.sim.process_frame_at_time(time.time() - self.t)
        # self.sim.process_frame_sync()
        """update_delta = 1/4
        if time.time() - self.t_update >= update_delta:
            self.t_update = time.time()
            self.sim.process_frame_at_time(time.time() - self.t)"""