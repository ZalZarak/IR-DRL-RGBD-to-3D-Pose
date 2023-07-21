#!/usr/bin/env python




import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import numpy as np
from time import process_time, time
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import PointCloud2
import sys
from modular_drl_env.gym_env.environment import ModularDRLEnv
from modular_drl_env.util.configparser import parse_config
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.world.obstacles.shapes import Box, Sphere
from modular_drl_env.planner.planner_implementations.rrt import BiRRT
import ros_numpy
from stable_baselines3 import PPO
import open3d as o3d
from time import sleep
import yaml
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

import pandas as pd

import torch
from collections import deque
from scipy.spatial import cKDTree

from collections import deque

import pybullet as pyb

import pickle

from voxelization import get_voxel_cluster, set_clusters, get_neighbouring_voxels_idx

# TODO: 
# farben bug fixen
# Custom Collision Check: Mindestens 3 Collisions bevor er wirklich abbricht
# RGB durch Voxels
# Statistical outlier removal
# Mutex/Farbenguard um Farbenbug zu ändern (solange er farben ändert, Rendern überspringen)


# TODO temporal filter:
# 1. create numpy array of size temporal horizon x (3D shape of voxel grid)
# 2. check by index wether position was occupied in the past

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class listener_node_one:
    def __init__(self, action_rate, control_rate, num_voxels, point_cloud_static):
        # yaml config file
        self.config_path = '/home/moga/Desktop/IR-DRL/Sim2Real/config_data/config.yaml'
        self.config = self.load_config(self.config_path)

        # variables for logging real clock time
        self.start_sec = None
        self.current_time = None     

        # overall mode, True=DRL, False=RRT, default is DRL
        self.mode = True 
        self.trajectory_idx = 0

        # robot data
        self.end_effector_link_id = 6
        self.end_effector_xyz = None
        self.end_effector_rpy = None
        self.end_effector_quat = None
        self.end_effector_xyz_sim = None
        #filtered steps in the simulation that are bigger than the distance threshhold
        self.sim_step = 0
        # filtered steps in the real environment that are bigger than the distance thresshold
        self.real_step = 0
        self.durations = []  # list of floats
        self.joints = None
        self.velocities = None
        self.effort = None
        self.goal = None
        self.q_goal = None
        self.trajectory = None
        self.drl_horizon = self.config['drl_horizon']
      
        self.max_inference_steps = self.config['max_inference_steps']
        self.running_inference = False  # flag for interference between DRL and symsinc to prevent both from running parallel
        self.point_cloud_static = point_cloud_static
        self.static_done = False
        self.num_voxels = num_voxels
        self.color_voxels = True # einfärben der Voxels
        self.control_mode = True   # True=pos, False=joints
        self.startup = True
        self.drl_success = False
        self.dist_threshold = self.config['dist_threshold']
        # Real steps in the simulation with no filter
        self.inference_steps = 0
        self.inference_done = False
        self.camera_calibration = False
        self.dont_voxelize = False
        self.use_gpu = True # torch.cuda.is_available()
        self.data_set_guard = False

        # cbGetPointcloud
        self.points_raw = None

        # cbPointcloudToPybullet
        self.camera_transform_to_pyb_origin = np.eye(4)
        config_rpy = np.array(self.config['camera_transform_to_pyb_origin']['rpy'])
        config_matrix = (R.from_euler('xyz', config_rpy, degrees = True)).as_matrix()
        self.camera_transform_to_pyb_origin[:3, :3] = config_matrix
        # self.camera_transform_to_pyb_origin[:3, 0] = np.array([-1, 0, 0])      # vektor für Kamera x achse in pybullet
        # self.camera_transform_to_pyb_origin[:3, 1] = np.array([0, 0, -1])      # vektor für Kamera y achse in pybullet
        # self.camera_transform_to_pyb_origin[:3, 2] = np.array([0, -1, 0])      # vektor für Kamera z achse in pybullet
        self.camera_transform_to_pyb_origin[:3, 3] = np.array(self.config['camera_transform_to_pyb_origin']['xyz'])
        self.pyb_to_camera = np.eye(4)
        self.pyb_to_camera[:3, :3] = config_matrix.T
        self.pyb_to_camera[:3, 3] = np.matmul(-(config_matrix.T), np.array(self.config['camera_transform_to_pyb_origin']['xyz']))
        
        # pre-allocate the rotation matrix on the gpu in case we use it
        if self.use_gpu:
            self.camera_transform_gpu = torch.from_numpy(self.camera_transform_to_pyb_origin).to('cuda')
        
        # boundaries for the point cloud, such that we constrain it into a statically determined box
        # useful for normalized clustering later on
        self.points_lower_bound = np.array([-0.5, -1, -0.1, 1])
        self.points_upper_bound = np.array([1.2, 1.25, 1.5, 1])       

        # we also rotate the bounds into camera space, such that we can apply them for filtering before rotating the raw point cloud
        tmp = np.matmul(self.pyb_to_camera, self.points_lower_bound)
        tmp1 = np.matmul(self.pyb_to_camera, self.points_upper_bound)
        self.points_lower_bound_camera = np.min([tmp, tmp1], axis=0)
        self.points_upper_bound_camera = np.max([tmp, tmp1], axis=0)
        if self.use_gpu:
            self.points_lower_bound_camera = torch.from_numpy(self.points_lower_bound_camera).to('cuda')
            self.points_upper_bound_camera = torch.from_numpy(self.points_upper_bound_camera).to('cuda')

        # voxelization attributes
        self.voxel_size = self.config['voxel_size']
        self.robot_voxel_safe_distance = self.config['robot_voxel_safe_distance']
        
        #change to 5 if dynamic obstacles are there
        self.inference_steps_per_pointcloud_update = 1
        
        
        #part for voxel clustering
        self.enable_clustering = False #enable or disable clustering for performance reasons
        self.robot_voxel_cluster_distance = 0.3 #TODO: optimize this
        self.neighbourhood_threshold = np.sqrt(2)*self.voxel_size + self.voxel_size/10
        self.voxel_cluster_threshold = self.config['voxel_cluster_threshold'] #TODO: Variabel an der anzahl von voxeln ändern nicht hardcoden
        self.voxel_centers = None
        #Optional for recording voxels
        #self.all_frames = [] 
        
        #self.recording_no = 1

        #Part for logging Sim2Real csv
        self.logging = True
        self.additional_info = dict()
        self.log = [] 
        
        #test = o3d.geometry.VoxelGrid.create_dense(np.array([0,0,0]), np.array([0,0,0]), 0.05, self.points_upper_bound[0]-self.points_lower_bound[0], self.points_upper_bound[1]-self.points_lower_bound[1], self.points_upper_bound[2]-self.points_lower_bound[2])
        #print("voxel menge", len(test.get_voxels()))
         

         ## Set for saving voxel positons and check
        self.voxel_positions = set()

        ##TODO: Create 5 second average of voxel positions at start, and add these to the voxel_positions set

        
        self.trajectory_client = actionlib.SimpleActionClient(
            "scaled_pos_joint_traj_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        timeout = rospy.Duration(5)
        if not self.trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        
        # custom sim2real config parsen
        #_, env_config = parse_config("/home/moga/Desktop/IR-DRL/configs/S2R/obstsensor_trajectory_PPO.yaml", False) #False = kein Training
        _, env_config = parse_config("/home/moga/Desktop/IR-DRL/configs/S2R/s2rexperiment_benno_config_voxels.yaml", False)
        env_config["env_id"] = 0
        # mit der config env starten
        self.env = ModularDRLEnv(env_config)
        self.virtual_robot = self.env.robots[0] # env initialisiert robot, wird für spätere Verwendung in der virtual_robot variable gespeichert
        #self.env.reset()  # taken out, replace by command below
        self.custom_env_reset()
        # now that the robot is there, we can get the action size
        self.actions = np.zeros((100, len(self.virtual_robot.all_joints_ids)))

        self.voxels = []        # voxel objects in pybullet simulation
        self.pos_nowhere= np.array([0,0,-100])
        # initialize probe_voxel and obstacle_voxels
        self.initialize_voxels()
       
        pyb_u.toggle_rendering(False)
        self.virtual_robot.goal.delete_visual_aux()
        pyb_u.toggle_rendering(True)

        # load DRL model
        #self.model = PPO.load("/home/moga/Desktop/IR-DRL/models/weights/model_interrupt.zip")  # trajectory
        self.model = PPO.load("/home/moga/Desktop/IR-DRL/models/weights/model_trained_voxels.zip")  # no trajectory
        
        # load RRT planner
        self.planner = BiRRT(self.virtual_robot, padding=False)
        # info: the next line is necessary because the planner automatically takes into account if a joint has been deactivated in the config
        # however, for sending commands to ros we need all six joints, that's why we manually overwrite the joint ids that the planner can see
        # such that it has all six
        self.planner.joint_ids = [pyb_u.pybullet_joints_ids[self.virtual_robot.object_id, joint_id] for joint_id in self.virtual_robot.all_joints_ids]
        self.planner.joint_ids_u = self.virtual_robot.all_joints_ids

        #optional: enable to see if there are differences between the env and model
        """from modular_drl_env.util.misc import analyse_obs_spaces
        analyse_obs_spaces(self.env.observation_space, self.model.observation_space)"""
        
        print("[Listener] Using GPU support for voxelization." if self.use_gpu else "[Listener] Using CPU for voxelization.")
        print("[Listener] Moving robot into resting pose")
        self._move_to_resting_pose()
        sleep(1)

        # init ros stuff
        print("[Listener] Started ee position callback")
        rospy.Subscriber("/tf", TFMessage, self.cbGetPos)
        sleep(1)
        print("[Listener] Started callback for joint angles")
        rospy.Subscriber("/joint_states", JointState, self.cbGetJoints)
        sleep(1)
        print("[Listener] Started callback for raw pointcloud data")
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.cbGetPointcloud)
        sleep(1)
        print("[Listener] Started callback for DRL inference")
        rospy.Timer(rospy.Duration(secs=1/action_rate), self.cbAction)
        print("[Listener] Started callback for Planner inference")
        rospy.Timer(rospy.Duration(secs=1/action_rate), self.cbActionPlanner)
        sleep(1)
        # rospy.Timer(rospy.Duration(secs=1/200), self.cbAction)
        rospy.Timer(rospy.Duration(secs=1/100), self.cbSimSync) # sync der Simulation mit dem echten Roboter so schnell wie möglich
        print("[Listener] Started callback for pointcloud voxelization")
        self.start_time = time() #for recording voxel purposes
        rospy.Timer(rospy.Duration(secs=1/240), self.cbPointcloudToPybullet)    
        sleep(1)
        print("[Listener] Started callback for controlling robot")
        rospy.Timer(rospy.Duration(secs=1/control_rate), self.cbControl)

        print("[Listener] initialized node")
        rospy.spin() #Lässt rospy immer weiter laufen

    def _move_to_resting_pose(self):
        # move robot to start position #TODO: solve through Training
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = JOINT_NAMES
        duration = rospy.Duration(3)  
        point = JointTrajectoryPoint()
        #point.positions = [i*np.pi/180 for i in [-81.25, -90, -90, 0, 0, 0]]
        point.positions = [i*np.pi/180 for i in [-180, -45, -90, -135, 90, 0]]
        point.time_from_start = duration
        goal.trajectory.points.append(point)
        self.trajectory_client.send_goal(goal) 
        self.trajectory_client.wait_for_result()

    # verwerten Daten, wandeln in das Format vom NN, fragen NN, wandeln Output vom NN in vom
    # UR5 Driver verstandene Commands
    # Frequenz: festgelegt von uns
    def cbAction(self, event):
        if self.joints is not None:  # wait until self.joints is written to for the first time
            if self.goal is not None and not self.inference_done and self.mode: # only do inference if there is a goal given by user and we're not already done with inference and we're not using a planner
                print("[cbAction] starting DRL inference")
                
                # set inference mutex such that other callbacks do nothing while new movement is calculated
                self.running_inference = True
                # manually reset a few env attributes (we don't want to use env.reset() because that would delete all the voxels)
                self.env.steps_current_episode = 0
                self.env.is_success = False

                self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids)
                # overwrite goal
                self.env.world.position_targets[0] = self.goal
                
                self.env.world.joints_targets[0] = self.q_goal[:5]  # TODO: might cause problems with unorthodox number/order of controlled joints
                self.virtual_robot.goal.on_env_reset(0)
                pyb_u.toggle_rendering(False)
                self.virtual_robot.goal.build_visual_aux()
                pyb_u.toggle_rendering(True)
                # reset sensors
                for sensor in self.env.sensors:
                    # sensor.reset()
                    sensor.update(0)
                # run inference
                obs = self.env._get_obs()
                pos_ee_last = self.virtual_robot.position_rotation_sensor.position
                self.inference_steps = 0
                
                while True:
                    # do nothing if an additional sim step would override parts of the trajectory that haven't been executed yet
                    if self.sim_step - self.real_step >= self.drl_horizon:
                        # Waiting for real_step to catch up to sim_step
                        continue

                    # code below is (maybe ?) deprecated, but might be necessary if some weird things happen with the pointcloud
                    # sync point cloud
                    # if not self.static_done and self.inference_steps % self.inference_steps_per_pointcloud_update == 0:
                    #     self.PointcloudToVoxel()
                    #     self.VoxelsToPybullet()
                    #     if self.point_cloud_static:
                    #         self.static_done = True
                            

                    action, _ = self.model.predict(obs, deterministic=True)
                    #info = logzeile für einen Step
                    obs, _, _, info = self.env.step(action)
                    self.inference_steps += 1
                    
                    # loop breaking conditions
                    if info["collision"]:
                        self.goal = None
                        self.actions = np.ones((100, len(self.virtual_robot.all_joints_ids))) * self.joints
                        self.sim_step = -1
                        self.running_inference = False
                        self.inference_done = True
                        self.env.episode += 1
                        # find the voxels that collide with the robot
                        self.dont_voxelize = True
                        pyb_u.toggle_rendering(False)
                        for tup in pyb_u.collisions:
                            if self.virtual_robot.object_id in tup and not self.probe_voxel.object_id in tup:
                                voxel_id = tup[0] if tup[0]!=self.virtual_robot.object_id else tup[1]
                                pyb.changeVisualShape(pyb_u.to_pb(voxel_id), -1, rgbaColor=[1, 0, 0, 1])
                        pyb_u.toggle_rendering(True)
                        print("[cbAction] Found collision during inference!")
                        self.dont_voxelize = False
                        return
                    elif info["is_success"]:
                        self.drl_success = True
                        self.actions[self.sim_step % self.actions.shape[0]], _ = pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)
                        self.sim_step = self.sim_step + 1
                        self.running_inference = False
                        self.inference_done = True
                        self.env.episode += 1
                        print("[cbAction] DRL inference successful")
                        return
                    elif self.inference_steps >= self.max_inference_steps:
                        print("[cbAction] Model isn't moving. Max iteration limit reached")
                        self.goal = None
                        self.inference_done = True
                        self.actions = np.ones((100, len(self.virtual_robot.all_joints_ids))) * self.joints
                        self.sim_step = -1
                        self.running_inference = False
                        self.env.episode += 1
                        return
                    # calc next step
                    pos_ee = self.virtual_robot.position_rotation_sensor.position
                    dist_diff = np.linalg.norm(pos_ee - pos_ee_last)
                    if dist_diff >= self.dist_threshold:
                        # self.actions.append(self.virtual_robot.joints_sensor.joints_angles)
                        self.actions[self.sim_step % self.actions.shape[0]], _ = pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)
                        self.sim_step = self.sim_step + 1
                        pos_ee_last = pos_ee  
                        print("[cbAction] Action added")

    #Planner for RRT                
    def cbActionPlanner(self, event):
        if self.joints is not None:
            if not self.mode and self.q_goal is not None and self.trajectory is not None and not self.inference_done:  # planner mode activated and a valid goal has been created in cbControl
                print("[cbAction] starting planner trajectory execution")
                # set inference mutex such that other callbacks do nothing while new movement is calculated
                self.running_inference = True
                # manually reset a few env attributes (we don't want to use env.reset() because that would delete all the voxels)
                self.env.steps_current_episode = 0
                self.env.is_success = False
                self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids)

                # reset sensors
                for sensor in self.env.sensors:
                    # sensor.reset()
                    sensor.update(0)

                # get one obs to correctly initialize a few values in the env internals
                _ = self.env._get_obs()

                while True:
                    if self.sim_step - self.real_step >= self.drl_horizon:
                        # Waiting for real_step to catch up to sim_step
                        continue

                    # turn angles into vector between -1 and 1 so that we can properly step the env in order to get logs at the end
                    action = self.trajectory[self.trajectory_idx]
                    action = ((action + self.virtual_robot._joints_limits_upper) / (self.virtual_robot._joints_range / 2)) - 1

                    _, _, _, info = self.env.step(action)

                    # loop breaking conditions
                    if info["collision"]:
                        self.q_goal = None
                        self.goal = None
                        self.trajectory = None
                        self.mode = True
                        self.virtual_robot.use_physics_sim = True
                        self.virtual_robot.control_mode = 2
                        self.actions = np.ones((100, len(self.virtual_robot.all_joints_ids))) * self.joints
                        self.sim_step = -1
                        self.running_inference = False
                        self.inference_done = True
                        self.env.episode += 1
                        # find the voxels that collide with the robot
                        pyb_u.toggle_rendering(False)
                        for tup in pyb_u.collisions:
                            if self.virtual_robot.object_id in tup and not self.probe_voxel.object_id in tup:
                                voxel_id = tup[0] if tup[0]!=self.virtual_robot.object_id else tup[1]
                                pyb.changeVisualShape(pyb_u.to_pb(voxel_id), -1, rgbaColor=[1, 0, 0, 1])
                        pyb_u.toggle_rendering(True)
                        print("[cbAction] Found collision during inference! Choose a new goal or try again.")
                        return
                    elif np.linalg.norm(self.trajectory[-1] - pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)[0]) < 8e-2:
                        self.mode = True
                        self.virtual_robot.use_physics_sim = True
                        self.virtual_robot.control_mode = 2
                        self.drl_success = True
                        self.actions[self.sim_step % self.actions.shape[0]], _ = pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)
                        self.sim_step = self.sim_step + 1
                        self.running_inference = False
                        self.inference_done = True
                        self.env.episode += 1
                        return

                    self.actions[self.sim_step % self.actions.shape[0]], _ = pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)
                    self.sim_step = self.sim_step + 1
                    print("[cbAction] Action added")
                    print("trajectory idx", self.trajectory_idx)
                    print("feheler", np.linalg.norm(self.trajectory[self.trajectory_idx] - pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)[0]))
                    # step up trajectory
                    if np.linalg.norm(self.trajectory[self.trajectory_idx] - pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)[0]) < 8e-2:
                        self.trajectory_idx += 1
                        
                        
    
    def cbControl(self, event):  
        if self.startup:
            pass # TODO   
        if self.goal is None and self.joints is not None:
            pyb_u.toggle_rendering(False)
            self.virtual_robot.goal.delete_visual_aux()
            pyb_u.toggle_rendering(True)
            
            self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids) #if last argument = none only 5 of the 6 joints get recognized
            print("[cbControl] current (virtual) position: " + str(self.end_effector_xyz_sim))
            print("[cbControl] current (virtual) joint angles: " + str(self.joints))  
            inp = input("[cbControl] Enter a goal by putting three float values (xyz) with a space between or \n[cbControl] (c) to calibrate camera position or\n[cbControl] (v) to adjust voxel size or\n[cbControl] (r) to return the robot into the starting configuration (WARNING: does not consider collisions, both real and virtual!) or\n[cbControl] (p) to initialize robot control via a sample based planer: \n")
            if len(inp) == 1:
                if inp == "r": # Resets the robot to the start position
                    print("[cbControl] Moving robot into resting pose")
                    self._move_to_resting_pose() 
                if inp[0] == "c": # calibrates the camera
                    self._control_calibrate()
                if inp[0] == "p":
                    self._control_planner()          # planner control callback ausgelagert  
                if inp[0] == "v": # voxelsize
                    self._control_voxelsettings()
            else:          
                inp = inp.split(" ")

                try:
                    inp = [float(ele) for ele in inp]
                except ValueError:
                    print("[cbControl] input in wrong format, try again!")
                    return
                # check if inverse kinematics can actually reach the xyz pos
                tmp_goal = np.array(inp)
                self.virtual_robot.position_rotation_sensor.update(0)
                self.virtual_robot.joints_sensor.update(0)
                # check for collision in starting position
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    print("[cbControl] current position of robot is in collision in simulation! Try again or check the camera/voxelization if the problem persists.")
                    return
                q_goal = self.virtual_robot._solve_ik(tmp_goal, None)
                self.virtual_robot.moveto_joints(q_goal, False, self.virtual_robot.all_joints_ids)
                self.virtual_robot.position_rotation_sensor.update(0)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    print("[cbControl] target position is in collision in simulation! Try again or check the camera/voxelization if the problem persists..")
                    choice = input("[cbControl] Do you still want to go ahead with your chosen target? (y/n)\n")
                    if choice == "no" or choice == "n":
                        return
                tmp_pos = self.virtual_robot.position_rotation_sensor.position
                self.sim_step = 0
                self.real_step = 0
                self.inference_steps = 0
                self.inference_done = False
                self.drl_success = False
                self.static_done = False
                if np.linalg.norm(tmp_pos - tmp_goal) > 5e-2:
                    print("[cbControl] could not find solution via inverse kinematics that is close enough, try another position")
                    choice = input("[cbControl] Do you still want to go ahead with your chosen target? (y/n)\n")
                    if choice == "no" or choice == "n":
                        return
                # safety: reset to actual joint angles before the other callback starts running (even though it does the same thing there too)
                self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids)
                # another safety: let a second pass to avoid bad consequences from concurrent callbacks overlapping
                sleep(1)
                self.goal = tmp_goal
                self.q_goal = q_goal
                # self.goal_sphere.position = self.goal
                
        else:
            # print("[cbControl] Starting trajectory transmission")
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = JOINT_NAMES
            
            # divide duration through action callback, the entire move should take around as many seconds as simulated
            fugde_factor = 0.04 # * self.env.sim_step # 0.02
            # self.durations = [(self.inference_steps * self.env.sim_step)/len(self.actions) + fugde_factor for _ in self.actions]
            v = 10e-2
            self.durations = [self.dist_threshold * (1/v) for _ in self.actions]
            
                  
            duration = 2 * self.dist_threshold * (1/v)
            # act = self.actions.pop(0)
            # act = self.actions[i]
            if self.sim_step < 0:
                return
            elif self.sim_step > self.real_step:
                print("[cbControl] Sending action")
                print("#"*20)
                # make step
                act = self.actions[self.real_step % self.actions.shape[0]]
                print("real_step :", self.real_step,"sim_step :", self.sim_step,  "act :", act, "inference_step: ", self.inference_steps)
                # TODO next step or sim_step - 1
                self.real_step = self.real_step + 1
                #print("[cbControl] joint angle difference:",np.linalg.norm(self.joints - act))
                point = JointTrajectoryPoint()
                point.positions = act
                #print(act)
                point.time_from_start = rospy.Duration(duration)
                goal.trajectory.points.append(point)
                # print("goal: ", goal)
                self.trajectory_client.send_goal(goal)
                #TODO: Every time a goal gets send to the real trajectory, data need to be logged for the real csv
                if self.logging == True: 
                    self.log_csv()

                # if not self.mode:  # in RRT/planner mode wait until the robot has progressed to the current waypoint
                #     self.trajectory_client.wait_for_result()
                # Ersatz für wait_for_result
                # TODO calibrate how long to wait for waypoint to publish, as these are diff in joint angles and not cartesian
                while np.linalg.norm(self.joints - act) > 5e-2:
                    sleep(0.01)

       
            
            if self.drl_success and self.sim_step == self.real_step: 
                self.goal = None
                self.q_goal = None
                self.trajectory = None
                self.drl_success = False
                self.inference_done = False
                print("[cbControl] Goal reached, Task failed successfully!")

    def _control_planner(self):
        # set mode to planner
        self.mode = False
        self.virtual_robot.use_physics_sim = False
        inp = input("[cbControl] Input six target joint angles with a space between each one: \n")
        inp = inp.split(" ")
        try:
            inp = np.array([float(ele) for ele in inp])
        except ValueError:
            print("[cbControl] input in wrong format, try again!")
            self.mode = True
            self.virtual_robot.use_physics_sim = True
            return
        # check for collision in current position
        self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        if pyb_u.collision:
            print("[cbControl] current position of robot is in collision in simulation! Try again or check the camera/voxelization if the problem persists.")
            self.mode = True
            self.virtual_robot.use_physics_sim = True
            return
        # check for collision in target position
        self.virtual_robot.moveto_joints(inp, False, self.virtual_robot.all_joints_ids)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        if pyb_u.collision:
            print("[cbControl] target position is in collision in simulation! Try again or check the camera/voxelization if the problem persists..")
            self.mode = True
            self.virtual_robot.use_physics_sim = True
            return
        # call the planner to plan a collision free route to the target
        self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids)
        self.trajectory = self.planner.plan(inp, self.env.world.active_objects)
        if self.trajectory is None:
            print("[cbControl] Planner failed, try again!")
            self.mode = True
            self.virtual_robot.use_physics_sim = True
            return
        print("trajectory:", len(self.trajectory))
        print("letzter eintrag trajectory:", self.trajectory[-1])
        self.virtual_robot.control_mode = 1
        self.sim_step = 0
        self.real_step = 0
        self.virtual_robot.world.position_targets[0] = np.array([1,2,3])
        self.trajectory_idx = 0
        self.q_goal = inp 
        self.goal = np.array([1,2,3])  # some nonsense goal to set the mutex 

    def _control_calibrate(self):
        was_static = self.point_cloud_static
        self.point_cloud_static = False
        self.camera_calibration = True                   
        print("[cbControl] current (virtual) camera position: " + str(self.camera_transform_to_pyb_origin[:3, 3]))
        print("[cbControl] current (virtual) camera rpy: " + str(self.config['camera_transform_to_pyb_origin']['rpy']))  
        inp = input("[cbControl] Enter a camera position: \n")
        inp = inp.split(" ")
        inp2 = input("[cbControl] Enter a camera rotation in extrinsic XYZ Euler format and in degrees: \n")
        inp2 = inp2.split(" ")
        try:
            inp = [float(ele) for ele in inp]
            inp2 = [float(ele) for ele in inp2]
        except ValueError:
            print("[cbControl] Input in wrong format, try again!")
            self.camera_calibration = False
            return
        # Update the config dictionary
        self.config['camera_transform_to_pyb_origin']['xyz'] = [float(value) for value in inp]
        self.config['camera_transform_to_pyb_origin']['rpy'] = [float(value) for value in inp2]
        # Save the updated configuration to the file
        self.save_config(self.config_path, self.config)
        # Reload the configuration to get the latest values
        self.config = self.load_config(self.config_path)

        self.camera_transform_to_pyb_origin[:3, :3] = (R.from_euler('xyz', np.array(inp2), degrees = True)).as_matrix()
        self.camera_transform_to_pyb_origin[:3, 3] = np.array(inp)
        if self.use_gpu:
            self.camera_transform_gpu = torch.from_numpy(self.camera_transform_to_pyb_origin).to('cuda')
        self.pyb_to_camera[:3, :3] = self.camera_transform_to_pyb_origin[:3, :3].T
        self.pyb_to_camera[:3, 3] = np.matmul(-(self.camera_transform_to_pyb_origin[:3, :3].T), np.array(inp))
        tmp = np.matmul(self.pyb_to_camera, self.points_lower_bound)
        tmp1 = np.matmul(self.pyb_to_camera, self.points_upper_bound)
        self.points_lower_bound_camera = np.min([tmp, tmp1], axis=0)
        self.points_upper_bound_camera = np.max([tmp, tmp1], axis=0)
        if self.use_gpu:
            self.points_lower_bound_camera = torch.from_numpy(self.points_lower_bound_camera).to('cuda')
            self.points_upper_bound_camera = torch.from_numpy(self.points_upper_bound_camera).to('cuda')
        self.camera_calibration = False
        self.point_cloud_static = was_static
        self.static_done = False

    def _control_voxelsettings(self):
        self.dont_voxelize = True
        print("[cbControl] Current voxel size: " + str(self.voxel_size))
        print("[cbControl] Current robot voxel safe distance: " + str(self.robot_voxel_safe_distance))
        print("[cbControl] Current minimal cluster size: " + str(self.voxel_cluster_threshold))
        print("[cbControl] Current number of voxels: " + str(self.num_voxels))
        inp = input("[cbControl] Enter new voxel size as a float:\n")
        try:
            val = float(inp)
        except ValueError:
            print("[cbControl] Invalid value for voxel size!")
        self.voxel_size = val
        self.config['voxel_size'] = val
        self.save_config(self.config_path, self.config)
        inp = input("[cbControl] Enter new robot voxel safe distance as a float:\n")
        try:
            val = float(inp)
        except ValueError:
            print("[cbControl] Invalid value for robot voxel safe distance!")
        self.robot_voxel_safe_distance = val
        self.config['robot_voxel_safe_distance'] = val
        self.save_config(self.config_path, self.config)
        inp = input("[cbControl] Enter a new minimal cluster size as an int:\n")
        try: 
            val = int(inp)
        except ValueError:
            print("[cbControl] Invalid value for cluster size!")
        self.voxel_cluster_threshold = val
        self.config['voxel_cluster_threshold'] = val
        self.save_config(self.config_path, self.config)
        inp = input("[cbControl] Enter a new number of voxels as an int:\n")
        try: 
            val = int(inp)
        except ValueError:
            print("[cbControl] Invalid value for number of voxels!")
        self.num_voxels = val
        self.delete_voxels()
        self.initialize_voxels()
        self.dont_voxelize = False
    
    def cbGetPointcloud(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.ones((np_data.shape[0], 4))
        if len(points) == 0:
            print("[cbGetPointcloud] No point cloud data received, check the camera and its ROS program!")
            return
        points[:, 0] = np_data['x']
        points[:, 1] = np_data['y']
        points[:, 2] = np_data['z']
        color_floats = np_data['rgb']  # float value that compresses color data
        # convert float values into 3-vector with RGB intensities, normalized to between 0 and 1
        color_floats = np.ascontiguousarray(color_floats)
        color_floats = (color_floats.view(dtype=np.uint8).reshape(color_floats.shape + (4,))[:,:3].astype(np.float64)) / 255
        
        # transfer data into class variables where other callbacks can get them
        self.data_set_guard = True
        self.colors = color_floats
        self.points_raw = points
        self.data_set_guard = False

    def cbPointcloudToPybullet(self, event):
        # callback for PointcloudToPybullet
        # static = only one update
        # dynamic = based on update frequency

        start = time()
        if self.points_raw is not None and not self.dont_voxelize: #and not self.running_inference:  # catch first time execution scheduling problems
            if self.point_cloud_static:
                if self.static_done:
                    return
                else:
                    self.static_done = True
                    self.PointcloudToVoxel()
                    #self.all_frames.append(self.voxel_centers.copy()) # Optional for recording
                    self.VoxelsToPybullet()
            else:
                self.PointcloudToVoxel()
                #print("[cb Pointcloud to Pybullet time: ]" , time() - start)
                #print("[cb Pointcloud to Pybullet FPS: ]" , 1/(time() - start))
                #self.all_frames.append(self.voxel_centers.copy()) # Optional for recording
                self.VoxelsToPybullet()
        #np.save("./voxels.npy", self.all_frames)
        #Optional for recording:
        """if time() - self.start_time > 20:
            with open('./voxels_' + str(self.recording_no) + '.pkl', 'wb') as outfile:
                pickle.dump(self.all_frames, outfile)
                
            self.recording_no += 1
            self.all_frames = []
            self.start_time = time()
            print("Dumped new frames")"""
        

    def PointcloudToVoxel(self):
        # if using GPU, points and colors are torch tensors
        # if using CPU, points and colors are numpy arrays
        if self.joints is not None and len(self.points_raw) != 0:
            while self.data_set_guard:
                sleep(0.0001)
            points = self.points_raw
            colors = self.colors
            
            if self.use_gpu:
                colors = torch.from_numpy(colors).to('cuda')
                points = torch.from_numpy(points).to('cuda')

            # cut off all points that are outside of the area of interest as defined by the user
            # we use the rotated boundaries to do this on the raw point data
            # this reduces the number of points drastically before we go to the costly rotation of them all
            lower_mask = (points >= self.points_lower_bound_camera).all(axis=1)
            upper_mask = (points <= self.points_upper_bound_camera).all(axis=1)
            if self.use_gpu:
                in_boundary_mask = torch.logical_and(lower_mask, upper_mask)
            else:
                in_boundary_mask = np.logical_and(lower_mask, upper_mask)
            points = points[in_boundary_mask]
            colors = colors[in_boundary_mask]

            # rotate raw points into PyBullet coordinate system
            if not self.use_gpu:
                ############## CPU #######################  
                points = np.dot(self.camera_transform_to_pyb_origin, points.T).T.reshape(-1,4)   
            else:
                #################### GPU ####################################
                # points = torch.from_numpy(points).to('cuda')
                # colors = torch.from_numpy(colors).to('cuda')

                points = torch.matmul(self.camera_transform_gpu, points.T).T.reshape(-1,4)

            # remove homogeneous component, don't need it anymore after the rotation  
            points = points[:, :3]  

            if not self.camera_calibration:
                
                # filter objects near endeffector
                ee_pos, _, _, _ = pyb_u.get_link_state(self.virtual_robot.object_id, self.virtual_robot.end_effector_link_id)
                # TODO may need to be adjusted with KINECT Cam
                if self.use_gpu:
                    mask_offset1 = torch.from_numpy(ee_pos) + torch.tensor([0.1, -0.1, 0.2])
                    mask_offset1 = mask_offset1.to('cuda')
                    mask_offset2 = torch.from_numpy(ee_pos) + torch.tensor([0, 0, 0.15])
                    mask_offset2 = mask_offset2.to('cuda')
                else:
                    mask_offset1 = ee_pos + np.array([0.1, -0.1, 0.2])
                    mask_offset2 = ee_pos + np.array([0, 0, 0.15])
                mask_left = self.delete_points_by_circle_center(points, mask_offset1)  # TODO: maybe find a general way to do this
                mask_above = self.delete_points_by_circle_center(points, mask_offset2)
                if self.use_gpu:
                    not_close_mask = torch.logical_and(mask_left, mask_above)
                else:
                    not_close_mask = np.logical_and(mask_left, mask_above)
                # filter by bool_array
                points = points[not_close_mask] 
                colors = colors[not_close_mask]

            # safety escape: the code below this will crash in case no points survive the filtering process
            if len(points) == 0:
                return
            
            # dump the results until now from the GPU
            if self.use_gpu:
                points = points.cpu().numpy()
                colors = colors.cpu().numpy()
            
            # get the offset for the voxel transformation later on

            # #scale all values for all axis between 0-1 to be able to determine static number of grids
            # points = points - self.points_lower_bound[:3]
            # points_range = self.points_upper_bound[:3] - self.points_lower_bound[:3]
            # points = points / (points_range)

            #points = np.append(points, [np.array([1, 1, 1]), np.array([0, 0, 0])], axis=0)

            pcd = o3d.geometry.PointCloud()
            # convert it into open 3d format
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors) # makes voxels have averaged colors of points, colors have to be normalize to between 0 and 1
            pcd = statistical_outlier_removal(pcd)
         
            #Downsample pointcloud: 
            #pcd = pcd.voxel_down_sample(voxel_size=0.05)
            
           

            #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=self.voxel_size, min_bound=self.points_lower_bound[:3], max_bound=self.points_upper_bound[:3])

            
            voxel_data = [(voxel.grid_index, voxel.color) for voxel in voxel_grid.get_voxels()]
            voxel_centers, voxel_colors = zip(*voxel_data)
            

            voxel_centers = np.array(voxel_centers)
            voxel_colors = np.array(voxel_colors)
            # Transform Voxel centers into xyz Coordinates            
            voxel_centers = voxel_centers * self.voxel_size + self.points_lower_bound[:3]# + offset_min

            if not self.camera_calibration:
                pyb_u.toggle_rendering(False)
                # move robot to real position, in case it's somewher else due to a planner or else
                joints_now, _ = pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)
                self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids)
                not_delete_mask = np.zeros(shape=(voxel_centers.shape[0],), dtype=bool)
                # TODO: parallelize this
                for idx, point in enumerate(voxel_centers):
                    pyb_u.set_base_pos_and_ori(self.probe_voxel.object_id, point, np.array([0, 0, 0, 1]))
                    self.probe_voxel.position = point
                    #checks if point is in close distance of the robot
                    query = pyb.getClosestPoints(pyb_u.to_pb(self.probe_voxel.object_id), pyb_u.to_pb(self.virtual_robot.object_id), self.robot_voxel_safe_distance)      
                    not_delete_mask[idx] = False if query else True
                voxel_centers = voxel_centers[not_delete_mask]
                voxel_colors = voxel_colors[not_delete_mask]
                # move robot back to where it was when filtering started
                self.virtual_robot.moveto_joints(joints_now, False, self.virtual_robot.all_joints_ids)
                pyb_u.toggle_rendering(True)
            pyb_u.set_base_pos_and_ori(self.probe_voxel.object_id, self.pos_nowhere, np.array([0, 0, 0, 1])) # TODO: Find out why this causes trouble, MOVES away probe voxel
            self.probe_voxel.position = self.pos_nowhere
            
            # Save voxels.npy
            #np.save("./voxels.npy",voxel_centers)
            #pickle.


            if self.enable_clustering:
            # get voxel_clusters
                voxel_clusters = get_voxel_cluster(voxel_centers, self.neighbourhood_threshold)
                # find clusters below a cluster size
                cluster_numbers, counts = np.unique(voxel_clusters, return_counts=True)
                # remove_cluster = []
                # for idx, clus in enumerate(cluster_numbers):
                #     if counts[idx] < self.voxel_cluster_threshold:
                #         remove_cluster.append(clus)
                include_cluster = cluster_numbers[np.invert(counts < self.voxel_cluster_threshold)]
                # remove voxels belonging to small clusters
                include_cluster_idx = np.isin(voxel_clusters, include_cluster)
                voxel_centers = voxel_centers[include_cluster_idx] 
                voxel_colors = voxel_colors[include_cluster_idx]
                

            self.voxel_centers = voxel_centers
            self.voxel_colors = voxel_colors

            

    def VoxelsToPybullet(self):
        if self.voxel_centers is not None:
            pyb_u.toggle_rendering(False)
            # update voxel positions
            for idx, voxel in enumerate(self.voxels):
                if idx >= len(self.voxel_centers):
                    # set all remaining voxels to nowhere
                    for i in range(idx, len(self.voxels)):
                        self.voxels[i].position = self.pos_nowhere
                        pyb_u.set_base_pos_and_ori(self.voxels[i].object_id, self.pos_nowhere, np.array([0, 0, 0, 1]))
                    break
                pyb_u.set_base_pos_and_ori(voxel.object_id, self.voxel_centers[idx], np.array([0, 0, 0, 1]))          
                voxel.position = self.voxel_centers[idx]
            # calculate new colors
            if self.color_voxels and len(self.voxel_centers) != 0:
                # voxel_norms = np.linalg.norm(self.voxel_centers - self.camera_transform_to_pyb_origin[:3, 3], axis=1)
                # max_norm, min_norm = np.max(voxel_norms), np.min(voxel_norms)
                # voxel_norms = (voxel_norms - min_norm) / (max_norm -  min_norm)
                # colors = np.ones((len(voxel_norms), 4))
                # colors = np.multiply(colors, voxel_norms.reshape(len(voxel_norms),1))
                # colors[:,3] = 1
                # colors[:,2] *= 0.5
                # colors[:,1] *= 0.333
                # colors[:,:3] = 1 - colors[:, :3]
                ones = np.ones((self.voxel_colors.shape[0], 1), dtype=np.float32)
                new_colors = np.concatenate([self.voxel_colors, ones], axis=1)
                for idx, voxel in enumerate(self.voxels):
                    if idx >= len(self.voxel_centers):
                        break
                    pyb.changeVisualShape(pyb_u.to_pb(voxel.object_id), -1, rgbaColor=new_colors[idx])
            pyb_u.toggle_rendering(True)
            
    
    def delete_points_by_circle_center(self, points, pos):
        # returns Boolean array or tensor
        if self.use_gpu:
            probe_norm = torch.norm((points - pos), dim=1)
        else:
            probe_norm = np.linalg.norm((points - pos), axis=1)
        return probe_norm > self.robot_voxel_safe_distance
    
    @staticmethod
    def pybullet_distance_check(voxel_id, qwe):
        pass


    #liest aktuelle Joints des real ur5 aus und überträgt sie in die Simulation
    def cbSimSync(self, event):
        if self.joints is not None and not self.running_inference:  # only move if joints are not None symsinc wird nur aufgerufen, wenn RRT aktuell nicht läuft
            self.virtual_robot.moveto_joints(self.joints, False, self.virtual_robot.all_joints_ids) #False = dont use physics sim 
        self.virtual_robot.position_rotation_sensor.update(0)
        self.end_effector_xyz_sim = self.virtual_robot.position_rotation_sensor.position

    # holen Daten, Frequenz: vom Sender (UR5)
    def cbGetPos(self, data):
        for entry in data.transforms:
            parent_frame_id = entry.header.frame_id 
            frame_id = entry.child_frame_id
            if parent_frame_id == "base" and frame_id == "tool0_controller":
                xyz = np.array([entry.transform.translation.x, entry.transform.translation.y, entry.transform.translation.z])
                quat = np.array([entry.transform.rotation.x, entry.transform.rotation.y, entry.transform.rotation.z, entry.transform.rotation.w])
                self.end_effector_xyz = xyz
                self.end_effector_quat = -quat #anpassung an rviz
                self.end_effector_rpy = pyb.getEulerFromQuaternion(self.end_effector_quat)
                
            
    def cbGetJoints(self, data):
        # sometimes the robot driver reports joint angles in an order different from the one we need
        # that's why we need to map the reported angles to our order that is given in JOINT_NAMES
        
        if self.start_sec is None:
            self.start_sec = data.header.stamp.secs + data.header.stamp.nsecs * (10**-9)
        
        pos_data = dict()
        val_data = dict()
        effort_data = dict()
        for idx, name in enumerate(data.name):
            pos_data[name] = data.position[idx]
            val_data[name] = data.velocity[idx]
            effort_data[name] = data.effort[idx]
            
        output_pos = []
        output_vel = []
        output_eff = []
        for name in JOINT_NAMES:
            output_pos.append(pos_data[name])
            output_vel.append(val_data[name])
            output_eff.append(effort_data[name])

        self.joints = np.array(output_pos, dtype=np.float32)

        #for logging
        self.current_time = (data.header.stamp.secs + data.header.stamp.nsecs * (10**-9)) - self.start_sec 
        
        self.velocities =np.array(output_vel,dtype=np.float32)
        self.effort = np.array(output_eff,dtype=np.float32)
        
    
    def initialize_voxels(self):
        # generate probe_voxel and obstacle_voxels
        pyb_u.toggle_rendering(False)
        #self.probe_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=[1, 1, 1, 0])
        self.probe_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], sim_step=self.env.sim_step, sim_steps_per_env_step=self.env.sim_steps_per_env_step, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=[1, 1, 1, 0])
        self.probe_voxel.build()
        # self.goal_sphere = Sphere(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[],radius = 0.5, move_step=0, color=[1, 1, 1, 0])
        for i in range(self.num_voxels):
            #new_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=np.concatenate((np.random.uniform(size=(3,)), np.ones(1))))
            new_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], sim_step=self.env.sim_step, sim_steps_per_env_step=self.env.sim_steps_per_env_step, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=[1, 0, 0, 1])
            self.voxels.append(new_voxel)
            new_voxel.build()
            self.env.world.obstacle_objects.append(new_voxel)
        self.env.world.active_objects = self.env.world.obstacle_objects
        pyb_u.toggle_rendering(True)

    def delete_voxels(self):
        pyb_u.toggle_rendering(False)
        for voxel in self.voxels:
            #initialise pybullet again
            pyb.removeBody(pyb_u.to_pb(voxel.object_id))
            del voxel
        self.env.world.obstacle_objects = []
        self.env.world.active_objects = []
        self.voxels = []
        pyb_u.toggle_rendering(True)

    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    
    def save_config(self, file_path, config):
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

    def custom_env_reset(self):
        # replaces a call to env reset
        # we do this to avoid a few problems that arise when using the env.reset method outside of the context of training
        # or inference without sim2real functionality
        self.env.steps_current_episode = 0
        self.env.sim_time = 0
        self.env.cpu_time = 0
        self.env.inference_time = 0
        self.env.cpu_reset_epoch = process_time()
        self.env.reward = 0
        self.env.reward_cumulative = 0
        self.env.log = []
        pyb_u.toggle_rendering(False)
        self.env.world.reset(0)
        pyb_u.toggle_rendering(True)
        self.env.active_robots = [True for _ in self.env.robots]
        for sensor in self.env.sensors:
            sensor.reset()

    #This logs all the important data, call at every real step
    def log_csv(self):
        #print("look here:", self.env.log[-1])
        #get joint velocities and real_effort from ros node
        sim_log = self.env.log[-1]
        

        
        additional_info = {
            "real_step":self.real_step,
            "real_joint_velocities": self.velocities,
            "real_effort": self.effort,
            "real_joint_positions": self.joints,
            "current_time" : self.current_time
        }

        #print("________________________ADD_INFO_______________")
        #print(additional_info)

        #merge both dicts
        info = {**sim_log, **additional_info}

        #Create CSV and add everything
        self.log.append(info)

        #TODO: only do this at the end not all the time
        try:
            pd.DataFrame(self.log).to_csv("./models/env_logs/episode_real_" + str(info["episodes"]) + ".csv")   
            pd.DataFrame(self.env.log).to_csv("./models/env_logs/episode_simulated_" + str(info["episodes"]) + ".csv")    
        except ValueError:
            print("error")
            columns = []
            for entry in self.env.log:
                if len(columns) == 0:
                    columns += entry.keys()
                else:
                    for key in entry.keys():
                        if key not in columns:
                            print(entry)
                            print(key)
            #print(self.env.log)

def statistical_outlier_removal(pcd, n_neighbors=20, std_ratio=2.0):
    points = np.asarray(pcd.points)  # Convert the points to a numpy array

    # Calculate the distances and indices using sklearn's NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(points)
    distances, indices = neigh.kneighbors(points)

    # Calculate mean and standard deviation
    mean_distances = np.mean(distances, axis=1)
    std_distances = np.std(distances, axis=1)

    # Identify points with a distance larger than mean + std_ratio * std_deviation
    outlier_mask = mean_distances > mean_distances.mean() + std_ratio * mean_distances.std()

    # Remove the outliers
    filtered_points = points[~outlier_mask]

    # Create a new PointCloud object for the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_colors = colors[~outlier_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd






if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True, disable_signals=True) 
    listener = listener_node_one(action_rate=60, control_rate=120, num_voxels=2000, point_cloud_static=False)




