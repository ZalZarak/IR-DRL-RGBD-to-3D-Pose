from .goal_implementations import *
from .goal import Goal
from .goal_implementations.peter_goal import PeterGoal
from .goal_implementations.peter_test_goal import PeterTestGoal


class GoalRegistry:
    _goal_classes = {}

    @classmethod
    def get(cls, goal_type:str) -> Goal:
        try:
            return cls._goal_classes[goal_type]
        except KeyError:
            raise ValueError(f"unknown goal type: {goal_type}")

    @classmethod
    def register(cls, goal_type:str):
        def inner_wrapper(wrapped_class):
            cls._goal_classes[goal_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

GoalRegistry.register('PositionCollision')(PositionCollisionGoal)
GoalRegistry.register('PositionRotationCollision')(PositionRotationCollisionGoal)
GoalRegistry.register('PositionRotationBetterSmoothingCollision')(PositionCollisionBetterSmoothingGoal)
GoalRegistry.register('PositionCollisionTrajectory')(PositionCollisionTrajectoryGoal)
GoalRegistry.register('PositionCollisionNoShaking')(PositionCollisionGoalNoShaking)
GoalRegistry.register('PositionCollisionNoShakingProximity')(PositionCollisionGoalNoShakingProximity)
GoalRegistry.register('PositionCollisionNoShakingProximityV2')(PositionCollisionGoalNoShakingProximityV2)
GoalRegistry.register('PositionCollisionNoShakingProximityV3')(PositionCollisionGoalNoShakingProximityV3)
GoalRegistry.register('PeterTestGoal')(PeterTestGoal)
GoalRegistry.register('PeterGoal')(PeterGoal)
