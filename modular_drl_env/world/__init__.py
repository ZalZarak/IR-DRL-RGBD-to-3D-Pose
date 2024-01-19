from .world_implementations import *
from .world import World
from .world_implementations.human_pose_world import HumanPoseWorld2, HumanPoseWorld


class WorldRegistry:
    _world_classes = {}

    @classmethod
    def get(cls, world_type:str) -> World:
        try:
            return cls._world_classes[world_type]
        except KeyError:
            raise ValueError(f"unknown world type for: {world_type}")

    @classmethod
    def register(cls, world_type:str):
        def inner_wrapper(wrapped_class):
            cls._world_classes[world_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

# Pybullet worlds
WorldRegistry.register('RandomObstacle')(RandomObstacleWorld)
WorldRegistry.register('Testcases')(TestcasesWorld)
WorldRegistry.register('TableExperiment')(TableExperiment)
WorldRegistry.register('Generated')(GeneratedWorld)
WorldRegistry.register('KukaShelfExperiment')(KukaShelfExperiment)
WorldRegistry.register('RandomBoxes')(RandomBoxesWorld)
WorldRegistry.register('AvoidObstacle')(AvoidObstacle)
WorldRegistry.register('PlateExperiment')(PlateExperiment)
WorldRegistry.register('S2RExperiment')(S2RExperiment)
WorldRegistry.register('S2RExperimentVoxels')(S2RExperimentVoxels)
WorldRegistry.register('HumanPoseWorld2')(HumanPoseWorld2)
WorldRegistry.register('HumanPoseWorld')(HumanPoseWorld)