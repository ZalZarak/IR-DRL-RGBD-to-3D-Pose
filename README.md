# Modular DLR Gym Env for Robots with PyBullet


<p float="left">
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/docs/gifs/GifReal.gif" width="400" />
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/docs/gifs/GifSim.gif" width="400" /> 
</p>


## Introduction

This repository provides a platform for training virtual agents in robotics tasks using Deep Reinforcement Learning (DRL). The code is built on OpenAI Gym, Stable Baselines, and PyBullet. The system is designed to operate using a modular approach, giving the user the freedom to combine, configure, and repurpose single components like goal, robot or sensor types.

An integral part of this project is the implementation of a transition mechanism from a simulated to a real-world environment. By leveraging the functionalty of ROS (Robot Operating System) and Voxelisation techniques with Open 3D, there is a system established that can effectively deploy trained models into real-world scenarios. There they are able to deal with static and dynamic obstacles.

This project is intended to serve as a resource for researchers, robotics enthusiasts, and professional developers interested in the application of Deep Reinforcement Learning in robotics.

## Getting Started

To get started with the project, please follow the instructions in the following sections:

- [Setup](./Setup.md): Instructions for setting up and installing the project.
- [Training/Evaluation](./Training_Evaluation.md): Information on how to train and evaluate the models.
- [Perception](./Perception.md): Details about the perception module of the project.
- [Deployment](./Deployment.md): Guidelines for deploying the project in a Real World environment.

Please ensure you read through all the sections to understand how to use the project effectively.

## RGBD-to-3D-Pose Intgration
This fork provides an example integration of [RGBD-to-3D-Pose](https://github.com/ZalZarak/RGBD-to-3D-Pose) into this repo. Last tested commit: f03256e

- Clone the project
- Clone RGBD-to-3D-Pose as submodule:   
  ``git submodule update --init --recursive``
- Follow installation of RGBD-to-3D-Pose: [https://github.com/ZalZarak/RGBD-to-3D-Pose](https://github.com/ZalZarak/RGBD-to-3D-Pose)
- Change the path of the RGBD-to-3D-Pose config under [modular_drl_env/RGBDto3Dpose/src/config.py](modular_drl_env/RGBDto3Dpose/src/config.py) to yours.
- An example is provided here: [configs/RGBDto3dpose-configs/rgbdto3dpose-config-for-irdrl.yaml](configs/RGBDto3dpose-configs/rgbdto3dpose-config-for-irdrl.yaml)

Example integrations are provided with those files:
- [configs/human_pose_config.yaml](configs/human_pose_config.yaml)
- [configs/human_pose_config_2.yaml](configs/human_pose_config_2.yaml)
- [modular_drl_env/world/world_implementations/human_pose_world.py](modular_drl_env/world/world_implementations/human_pose_world.py)
- [modular_drl_env/goal/goal_implementations/human_pose_goal.py](modular_drl_env/goal/goal_implementations/human_pose_goal.py)

Preperation:   

Create a simulator object, with your ``**your_config`` as input.   
Config explanation: [modular_drl_env/RGBDto3Dpose/config/config_explanation.yaml](modular_drl_env/RGBDto3Dpose/config/config_explanation.yaml)

Human Geometry needs to generated. This can be done manually, as in ``HumanPoseWorld``.   
Then the pybullet object ids need to be added to ``simulator.limbs_pb`` at the correct place.
Then in RGBDto3DPose config ``build_geometry`` must be ``false``.

Or Simulator can do the generation, as in ``HumanPoseWorld2``.   
Then ``pyb_u.pybullet_object_ids`` and ``pyb_u.gym_env_str_names`` need to be filled correctly.
``build_geometry`` must be ``true``.

If Simulator runs in real-time (with the perception module), you can move the limbs with ``simulator.process_frame_sync()``

If Simulator reads the joints from a file, you can move with ``simulator.process_frame_at_time()``.   
This takes a timestamp as input and moves the limbs to the positions defined in the closest next time stamp.
You need to figure out how to calculate the current simulated time. ``HumanPoseWorld`` does it but requires a 
speedup factor to show it in normal speed. This is an IR-DRL issue.

Also, you can prevent Simulator from doing its own pybullet simulation steps by setting ``do_sim_steps`` to ``False``.
Then you need to call ``pybullet.stepSimulation()`` by your own. ``reset_limb_velocities`` needs to be called manually 
after each simulation step where limbs where moved. For efficient simulation, call 
``p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(b))`` with ``b=False`` before the step and with ``b=True`` after the step.   
No example is implemented, but you can orientate at the Simulator Function ``step_full``.

The current joint positions are available under simulator.joints. 25th position is background. 
See RGBDto3DPose config for details. 

You may do something like 
```
try:
    ret["joints"] = self.robot.world.sim.joints[:25, :].copy()
except AttributeError and TypeError:
    ret["joints"] = np.zeros([24, 3])
```
in ``get_observation`` method of your goal. I couldn't figure out how to make IR-DRL accept it.
The Error should only arise during the first call, as world is not initialized yet.

If geometry is generated with IR-DRL-tools, obstacles might be observed automatically, though I give you no guarantee for that.

