import numpy as np

from modular_drl_env.world import World
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.shapes import Box


class PeterTestWorld(World):
    def set_up(self):
        self.ground_plate = GroundPlate(True)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -1]))

        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5], seen_by_obstacle_sensor=False)
        self.table.build()


    def reset(self, success_rate: float):
        #self.robot.reset()
        pass

    def update(self):
        pass