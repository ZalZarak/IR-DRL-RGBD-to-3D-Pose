import gym
import numpy as np
import pybullet as pyb
from time import process_time
import pandas as pd
import os
import platform

# import abstracts
from modular_drl_env.robot.robot import Robot
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.goal.goal import Goal
from modular_drl_env.world.world import World

# import pybullet wrapper
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

# import implementations, new ones hav to be added to the registries to work
#   worlds
from modular_drl_env.world import WorldRegistry
#   robots
from modular_drl_env.robot import RobotRegistry
#   sensors
from modular_drl_env.sensor import SensorRegistry
#   goals
from modular_drl_env.goal import GoalRegistry

class ModularDRLEnv(gym.Env):

    def __init__(self, env_config):
        
        #   general env attributes
        # run mode
        self.train = env_config["train"]
        # flag for normalizing observations
        self.normalize_observations = env_config["normalize_observations"]
        # flag for normalizing rewards
        self.normalize_rewards = env_config["normalize_rewards"]
        # flag for rendering auxillary geometry spawned by the scenario
        self.show_auxillary_geometry_world = env_config["show_world_aux"]
        # flag for rendering auxillary geometry spawned by the goals
        self.show_auxillary_geometry_goal = env_config["show_goal_aux"]
        # flag for rendering auxillary geometry spawned by the sensors
        self.show_auxillary_geometry_sensors = env_config["show_sensor_aux"]
        # maximum steps in an episode before timeout
        self.max_steps_per_episode = env_config["max_steps_per_episode"]
        # number of episodes after which the code will exit on its own, if set to -1 will continue indefinitely until stopped from the outside
        self.max_episodes = env_config["max_episodes"]  
        # 0: no logging, 1: logging for console every episode, 2: logging for console every episode and to csv after maximum number of episodes has been reached or after every episode if max_episodes is -1
        # 3: same as 2 but will also log the position, orientation and shape of every obstacle as well as the closest distances to those for each robot (performance heavy)
        self.logging = env_config["logging"] 
        # length of the stat arrays in terms of episodes over which the average will be drawn for logging
        self.stat_buffer_size = env_config["stat_buffer_size"]  
        # whether to use physics-free setting of object positions or actually let sim time pass in simulation
        self.use_physics_sim = env_config["use_physics_sim"]  
        # when using the physics sim, this is the amount of steps that we let pass per env step
        # the lower this value, the more often observations will be collected and a new action be calculated by the agent
        # note: if use_physics_sim is False, this does not affect anything other than the sim time counted by the gym env
        self.sim_steps_per_env_step = env_config["sim_steps_per_env_step"]
        # in seconds -> inverse is frame rate in Hz
        self.sim_step = env_config["sim_step"]  
        # the env id, this is used to recognize this env when running multiple in parallel, is used for some file related stuff
        self.env_id = env_config["env_id"]
        

        # tracking variables
        self.episode = 0
        self.steps_current_episode = 0
        self.sim_time = 0
        self.cpu_time = 0
        self.cpu_epoch = process_time()
        self.inference_time = 0
        self.inference_epoch = 0
        self.log = []
        # init and fill the stats with a few entries to make early iterations a bit more robust
        self.success_stat = [False, False]
        self.out_of_bounds_stat = [False, False]
        self.timeout_stat = [False, False]
        self.collision_stat = [False, False]
        self.cumulated_rewards_stat = [0]
        self.goal_metrics = []
        self.reward = 0
        self.reward_cumulative = 0

        #   engine setup

        #   asset path slicing
        # to access our assets, we need to direct the code towards the location within the python installation that we're in
        # or if this was downloaded as a repo, simply the neighboring assets folder
        # in both cases we use some os commands to get the correct folder path 
        assets_path = os.path.normpath(__file__)  # path of this file
        assets_path = assets_path.split(os.sep)  # split along os specific separator
        assets_path[-2] = "assets"  # replace second to last entry, which should be gym_env, with assets
        # stitch the path together again, leaving the last element, environment.py, out, such that this path is the correct asset path 
        if any(platform.win32_ver()):  # check for windows
            self.assets_path = os.path.join(assets_path[0], os.sep, *assets_path[1:-1])  # windows drive letter needs babysitting
        else:
            self.assets_path = os.path.join(os.sep, *assets_path[:-1])  
        
        # init pybullet from config
        pyb_u.init(self.assets_path, env_config["display"], env_config["sim_step"], env_config["gravity"])

        pyb_u.toggle_rendering(False)
        # init world from config
        self._world_setup(env_config)

        # init robots and their associated sensors and goals from config
        self.robots:list[Robot] = []
        self.sensors:list[Sensor] = []
        self.goals:list[Goal] = []
        self._robot_setup(env_config)
        pyb_u.toggle_rendering(True)

        # construct observation space from sensors and goals
        # each sensor and goal will add elements to the observation space with fitting names
        observation_space_dict = dict()
        for sensor in self.sensors:
            if sensor.add_to_observation_space:
                observation_space_dict = {**observation_space_dict, **sensor.get_observation_space_element()}  # merges the two dicts
        for goal in self.goals:
            if goal.add_to_observation_space:
                observation_space_dict = {**observation_space_dict, **goal.get_observation_space_element()}

        self.observation_space = gym.spaces.Dict(observation_space_dict)

        # construct action space from robots
        # the action space will be a vector with the length of all robot's control dimensions added up
        # e.g. if one robot needs 4 values for its control and another 6,
        # the action space will be a 10-vector with the first 4 elements working for robot 1 and the last 6 for robot 2
        self.action_space_dims = []
        for idx, robot in enumerate(self.robots):
            joints_dims, ik_dims = robot.get_action_space_dims()
            if robot.control_mode:  # aka if self.control_mode[idx] == 1 or == 2
                self.action_space_dims.append(joints_dims)
            else:  # == 0
                self.action_space_dims.append(ik_dims)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(sum(self.action_space_dims),), dtype=np.float32)

    def reset(self):
        pyb_u.toggle_rendering(False)
        # end execution if max episodes is reached
        if self.max_episodes != -1 and self.episode >= self.max_episodes:
            exit(0)

        # reset the tracking variables
        self.steps_current_episode = 0
        self.sim_time = 0
        self.cpu_time = 0
        self.inference_time = 0
        self.cpu_reset_epoch = process_time()
        self.reward = 0
        self.reward_cumulative = 0
        self.episode += 1
        if self.max_episodes == -1:  # if we have a finite amount of episodes, we want the log to hold everything, otherwise flush it for the next one
            self.log = []  

        # toggle pybullet rendering off
        
        self.world.reset(np.average(self.success_stat))

        # set all robots to active
        self.active_robots = [True for robot in self.robots]

        # reset the sensors to start settings
        for sensor in self.sensors:
            sensor.reset()

        # call the goals' update routine and get their metrics, if they exist
        self.goal_metrics = []
        for goal in self.goals:
            self.goal_metrics += goal.on_env_reset(np.average(self.success_stat))

        # render non-essential visual stuff
        if not self.train:
            if self.show_auxillary_geometry_goal:
                for goal in self.goals:
                    goal.delete_visual_aux()
                    goal.build_visual_aux()
            if self.show_auxillary_geometry_sensors:
                for sensor in self.sensors:
                    sensor.delete_visual_aux()
                    sensor.build_visual_aux()

        pyb_u.toggle_rendering(True)
        return self._get_obs()

    def _get_obs(self):
        obs_dict = dict()
        # get the sensor data
        for sensor in self.sensors:
            if sensor.add_to_observation_space:
                obs_dict = {**obs_dict, **sensor.get_observation()}
        for goal in self.goals:
            if goal.add_to_observation_space:
                obs_dict = {**obs_dict, **goal.get_observation()}

        # no normalizing here, that should be handled by the sensors and goals

        # as this is the last method called before the DRL agent steps in, we can measure the
        # inference time here and in step
        self.inference_epoch = process_time()

        return obs_dict

    def step(self, action):

        # measure inference time via the epoch, see above
        self.inference_time += process_time() - self.inference_epoch
        
        if self.steps_current_episode == 0:
            self.cpu_epoch = process_time()

        # convert to numpy
        action = np.array(action)
        
        # update world
        self.world.update()

        # apply the action to all robots that have to be moved
        action_offset = 0  # the offset at which the ith robot sits in the action array
        exec_times_cpu = []  # track execution times
        for idx, robot in enumerate(self.robots):
            if not self.active_robots[idx]:
                action_offset += self.action_space_dims[idx]
                exec_times_cpu.append(0)
                continue
            # get the slice of the action vector that belongs to the current robot
            current_robot_action = action[action_offset : self.action_space_dims[idx] + action_offset]
            action_offset += self.action_space_dims[idx]
            exec_time = robot.process_action(current_robot_action)
            # let simulated physical time pass
            for i in range(self.sim_steps_per_env_step):
                pyb.stepSimulation()
                self.sim_time += self.sim_step
            exec_times_cpu.append(exec_time)

        # update the sensor data
        for sensor in self.sensors:
            sensor.update(self.steps_current_episode)

        # update the collision model if necessary
        if not self.use_physics_sim:
            pyb_u.perform_collision_check()
        # run our collision handling
        pyb_u.get_collisions()

        # calculate reward and get termination conditions
        rewards = []
        dones = []
        successes = []
        timeouts = []
        oobs = []
        action_offset = 0
        for idx, robot in enumerate(self.robots):
            goal = robot.goal
            # only go trough calculations if robot has a goal and it is active
            if goal is not None and self.active_robots[idx]:
                # again get the slice of the entire action vector that belongs to the robot/goal in question
                current_robot_action = action[action_offset : self.action_space_dims[idx] + action_offset]
                # get reward of goal
                reward_info = goal.reward(self.steps_current_episode, current_robot_action)  # tuple: reward, success, done, timeout, out_of_bounds
                rewards.append(reward_info[0])
                successes.append(reward_info[1])
                # set respective robot to inactive after success, if needed
                if reward_info[1] and not goal.continue_after_success:
                    self.active_robots[idx] = False
                dones.append(reward_info[2] if not reward_info[1] else False)  # if the goal sends a success signal, we discard it's done signal to allow other robots to continue working; in single robot envs the overall done is still send, see below
                timeouts.append(reward_info[3])
                oobs.append(reward_info[4])
            action_offset += self.action_space_dims[idx]

        # determine overall env termination condition
        collision = pyb_u.collision
        is_success = np.all(successes)  # all goals must be succesful for the entire env to be
        done = np.any(dones) or collision or is_success  # one done out of all goals/robots suffices for the entire env to be done or anything collided or everything is successful
        timeout = np.any(timeouts)
        out_of_bounds = np.any(oobs)

        # reward
        # if we are normalizing the reward, we must also account for the number of robots 
        # (each goal will output a reward from -1 to 1, so e.g. three robots would have a cumulative reward range from -3 to 3)
        if self.normalize_rewards:
            self.reward = np.average(rewards)
        # otherwise we can just add the single rewards up
        else:
            self.reward = np.sum(rewards)
        self.reward_cumulative += self.reward

        # visual help, if enabled
        if self.show_auxillary_geometry_sensors:
            pyb_u.toggle_rendering(False)
            for sensor in self.sensors:
                sensor.delete_visual_aux()
                sensor.build_visual_aux()
            pyb_u.toggle_rendering(True)

        # update tracking variables and stats
        self.cpu_time = process_time() - self.cpu_epoch
        self.steps_current_episode += 1
        if done:
            self.success_stat.append(is_success)
            if len(self.success_stat) > self.stat_buffer_size:
                self.success_stat.pop(0)
            self.timeout_stat.append(timeout)
            if len(self.timeout_stat) > self.stat_buffer_size:
                self.timeout_stat.pop(0)
            self.out_of_bounds_stat.append(out_of_bounds)
            if len(self.out_of_bounds_stat) > self.stat_buffer_size:
                self.out_of_bounds_stat.pop(0)
            self.collision_stat.append(collision)
            if len(self.collision_stat) > self.stat_buffer_size:
                self.collision_stat.pop(0)
            self.cumulated_rewards_stat.append(self.reward_cumulative)
            if len(self.cumulated_rewards_stat) > self.stat_buffer_size:
                self.cumulated_rewards_stat.pop(0)

        # handle logging
        if self.logging == 0:
            # no logging
            info = {}
        if self.logging == 1 or self.logging == 2 or self.logging == 3:
            # logging to console or textfile

            # start log dict with env wide information
            info = {"env_id": self.env_id,
                    "episodes": self.episode,
                    "is_success": is_success, 
                    "collision": collision,
                    "timeout": timeout,
                    "out_of_bounds": out_of_bounds,
                    "step": self.steps_current_episode,
                    "success_rate": np.average(self.success_stat),
                    "out_of_bounds_rate": np.average(self.out_of_bounds_stat),
                    "timeout_rate": np.average(self.timeout_stat),
                    "collision_rate": np.average(self.collision_stat),
                    "cumulated_rewards": np.average(self.cumulated_rewards_stat),
                    "sim_time": self.sim_time,
                    "cpu_time_steps": self.cpu_time,
                    "cpu_time_full": self.cpu_time + self.cpu_epoch - self.cpu_reset_epoch,
                    "inference_time": self.inference_time}
            # get robot execution times
            for idx, robot in enumerate(self.robots):
                if not self.active_robots[idx]:
                    continue
                info["action_cpu_time_" + robot.name] = exec_times_cpu[idx] 
            # get the planner times, if there are any
            for idx, robot in enumerate(self.robots):
                planner_time = pyb_u.get_planner_time(robot.object_id)
                if planner_time:  # is 0 if no planner has planned on this robot
                    info["planner_time_" + robot.name] = planner_time
            # get the log data from sensors
            for sensor in self.sensors:
                if sensor.add_to_logging:
                    info = {**info, **sensor.get_data_for_logging()}
            # get log data from goals
            for goal in self.goals:
                if goal.add_to_logging:
                    info = {**info, **goal.get_data_for_logging()}
            # get log data from world
            if self.logging == 3:
                info = {**info, **self.world.get_data_for_logging()}

            self.log.append(info)

            # on episode end:
            if done:
                # write to console
                info_string = self._get_info_string(info)
                print(info_string)
                # write to textfile, in this case the entire log so far
                if self.logging == 2 or self.logging == 3:
                    if self.max_episodes == -1 or self.episode == self.max_episodes:
                        pd.DataFrame(self.log).to_csv("./models/env_logs/episode_" + str(self.episode) + ".csv")

        return self._get_obs(), self.reward, done, info

    ###################
    # utility methods #
    ###################

    def _get_info_string(self, info):
        """
        Handles writing info from sensors and goals to console. Also deals with various datatypes and should be updated
        if a new one appears in the code somewhere.
        """
        info_string = ""
        for key in info:
            # handle a few common datatypes and special cases
            if type(info[key]) == np.ndarray:
                to_print = ""
                for ele in info[key]:
                    to_print += str(round(ele, 3)) + " "
                to_print = to_print[:-1]  # cut off the last space
            elif type(info[key]) == np.bool_ or type(info[key]) == bool:
                to_print = str(int(info[key]))
            elif "time" in key and not "timeout" in key:
                if info[key] > 0.001:  # time not very small
                    to_print = str(round(info[key], 3))
                else:  # time very small
                    to_print = "{:.2e}".format(info[key])
            elif type(info[key]) == list:
                pass
            else:
                to_print = str(round(info[key], 3))
            info_string += key + ": " + to_print + ", "
        return info_string[:-1]  # cut off last space

    def manual_control(self, joints=False):
        """
        Debug method for controlling the robot at position 0 in the self.robots list.
        """
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 1)
        # code to manually control the robot in real time
        if not joints:
            roll = pyb.addUserDebugParameter("r", -4.0, 4.0, 0)
            pitch = pyb.addUserDebugParameter("p", -4.0, 4.0, 0)
            yaw = pyb.addUserDebugParameter("y", -4.0, 4.0, 0)
            fwdxId = pyb.addUserDebugParameter("fwd_x", -4, 8, 0)
            fwdyId = pyb.addUserDebugParameter("fwd_y", -4, 8, 0)
            fwdzId = pyb.addUserDebugParameter("fwd_z", -1, 3, 0)
            x_base = 0
            y_base = 0

            pyb.addUserDebugLine([0,0,0],[0,0,1],[0,0,1],parentObjectUniqueId=pyb_u.to_pb(self.robots[0].object_id), parentLinkIndex= pyb_u.pybullet_link_ids[self.robots[0].object_id ,self.robots[0].end_effector_link_id])
            pyb.addUserDebugLine([0,0,0],[0,1,0],[0,1,0],parentObjectUniqueId=pyb_u.to_pb(self.robots[0].object_id), parentLinkIndex= pyb_u.pybullet_link_ids[self.robots[0].object_id ,self.robots[0].end_effector_link_id])
            pyb.addUserDebugLine([0,0,0],[1,0,0],[1,0,0],parentObjectUniqueId=pyb_u.to_pb(self.robots[0].object_id), parentLinkIndex= pyb_u.pybullet_link_ids[self.robots[0].object_id ,self.robots[0].end_effector_link_id])

            lineID = 0
            textID = 0

            while True:
                if lineID:
                    pyb.removeUserDebugItem(lineID)
                    pyb.removeUserDebugItem(textID)

                # read inputs from GUI
                qrr = pyb.readUserDebugParameter(roll)
                qpr = pyb.readUserDebugParameter(pitch)
                qyr = pyb.readUserDebugParameter(yaw)
                x = pyb.readUserDebugParameter(fwdxId)
                y = pyb.readUserDebugParameter(fwdyId)
                z = pyb.readUserDebugParameter(fwdzId)
                oldxbase = x_base
                oldybase = y_base

                # build quaternion from user input
                command_quat = pyb.getQuaternionFromEuler([qrr,qpr,qyr])

                self.robots[0].moveto_xyzquat(np.array([x,y,z]),np.array(command_quat), self.use_physics_sim)
                if self.use_physics_sim:
                    for i in range(self.sim_steps_per_env_step):
                        pyb.stepSimulation()

                self.robots[0].position_rotation_sensor.reset()
                self.robots[0].joints_sensor.reset()
                pos = self.robots[0].position_rotation_sensor.position

                lineID = pyb.addUserDebugLine([x,y,z], pos.tolist(), [0,0,0])
                joints_str = ' '.join([str(round(ele, 2)) for ele in self.robots[0].joints_sensor.joints_angles])
                textID = pyb.addUserDebugText(joints_str, pos)
        else:
            q0 = pyb.addUserDebugParameter("q0", -np.pi, np.pi, 0)
            q1 = pyb.addUserDebugParameter("q1", -np.pi, np.pi, 0)
            q2 = pyb.addUserDebugParameter("q2", -np.pi, np.pi, 0)
            q3 = pyb.addUserDebugParameter("q3", -np.pi, np.pi, 0)
            q4 = pyb.addUserDebugParameter("q4", -np.pi, np.pi, 0)
            q5 = pyb.addUserDebugParameter("q5", -np.pi, np.pi, 0)
            
            textID = 0
            while True:
                if textID:
                    pyb.removeUserDebugItem(textID)
                q0r = pyb.readUserDebugParameter(q0)
                q1r = pyb.readUserDebugParameter(q1)
                q2r = pyb.readUserDebugParameter(q2)
                q3r = pyb.readUserDebugParameter(q3)
                q4r = pyb.readUserDebugParameter(q4)
                q5r = pyb.readUserDebugParameter(q5)

                command_quat = np.array([q0r, q1r, q2r, q3r, q4r, q5r])
                self.robots[0].moveto_joints(command_quat, self.use_physics_sim, self.robots[0].all_joints_ids)
                if self.use_physics_sim:
                    for i in range(self.sim_steps_per_env_step):
                        pyb.stepSimulation()

                self.robots[0].joints_sensor.reset()
                self.robots[0].position_rotation_sensor.reset()

                pos_text = ' '.join([str(round(ele, 3)) for ele in self.robots[0].position_rotation_sensor.position])

                textID = pyb.addUserDebugText(pos_text, self.robots[0].position_rotation_sensor.position)

    ####################
    # callback methods #
    ####################

    def set_goal_metric(self, name, value):
        """
        This method is only called from the outside by the custom logging callback (see callbacks/callbacks.py).
        It will change the goal metrics depending on the criteria defined by each goal.
        """
        # find all goals that have a metric with name
        for goal in self.goals:
            for metric in goal.metric_names:
                if metric == name:
                    setattr(goal, name, value)  # very bad for performance, but we'll never use so many goals that this will become relevant

    #################
    # setup methods #
    #################

    def _world_setup(self, env_config):
        world_type = env_config["world"]["type"]
        world_config = env_config["world"]["config"]
        world_config["assets_path"] = self.assets_path
        world_config["env_id"] = self.env_id
        world_config["sim_step"] = self.sim_step
        world_config["sim_steps_per_env_step"] = self.sim_steps_per_env_step
        
        self.world:World = WorldRegistry.get(world_type)(**world_config)
        self.world.set_up()
        if not self.train and self.show_auxillary_geometry_world:
            self.world.build_visual_aux()

    def _robot_setup(self, env_config):
        id_counter = 0
        for robo_entry in env_config["robots"]:
            robo_type = robo_entry["type"]
            robo_config = robo_entry["config"]
            # add some necessary attributes
            robo_config["id_num"] = id_counter
            robo_config["use_physics_sim"] = self.use_physics_sim
            robo_config["world"] = self.world
            robo_config["sim_step"] = self.sim_step
            id_counter += 1
            robot:Robot = RobotRegistry.get(robo_type)(**robo_config)
            self.robots.append(robot)
            robot.build()

            # create the two mandatory sensors
            if "report_joint_velocities" in robo_entry:
                jv = robo_entry["report_joint_velocities"]
            else:
                jv = False
            joint_sens_config = {"normalize": self.normalize_observations, "add_to_observation_space": True, 
                                 "add_to_logging": True, "sim_step": self.sim_step, "update_steps": 1, "sim_steps_per_env_step": self.sim_steps_per_env_step, "robot": robot, "add_joint_velocities": jv}
            posrot_sens_config = {"normalize": self.normalize_observations, "add_to_observation_space": False, 
                                 "add_to_logging": True, "sim_step": self.sim_step, "update_steps": 1, "sim_steps_per_env_step": self.sim_steps_per_env_step, "robot": robot,
                                 "link_id": robot.end_effector_link_id, "quaternion": True}
            new_rob_joints_sensor = SensorRegistry.get("Joints")(**joint_sens_config)
            new_rob_posrot_sensor = SensorRegistry.get("PositionRotation")(**posrot_sens_config)
            robot.set_joint_sensor(new_rob_joints_sensor)
            robot.set_position_rotation_sensor(new_rob_posrot_sensor)
            self.sensors.append(new_rob_posrot_sensor)
            self.sensors.append(new_rob_joints_sensor)

            # create the sensors indicated by the config
            if "sensors" in robo_entry:
                for sensor_entry in robo_entry["sensors"]:
                    sensor_type = sensor_entry["type"]
                    sensor_config = sensor_entry["config"]
                    sensor_config["sim_step"] = self.sim_step
                    sensor_config["robot"] = robot
                    sensor_config["normalize"] = self.normalize_observations
                    sensor_config["sim_steps_per_env_step"] = self.sim_steps_per_env_step
                    # deal with robot bound sensors that refer to other robots
                    if "target_robot" in sensor_config:
                        # go through the list of existing robots
                        for other_robot in self.robots:
                            # find the one whose name matches the target
                            if other_robot.name == sensor_config["target_robot"]:
                                sensor_config["target_robot"] = other_robot
                                break
                    new_sensor:Sensor = SensorRegistry.get(sensor_type)(**sensor_config)
                    self.sensors.append(new_sensor)
                    robot.sensors.append(new_sensor)
            
            if "goal" in robo_entry:
                # create the goal indicated by the config
                goal_type = robo_entry["goal"]["type"]
                goal_config = robo_entry["goal"]["config"]
                goal_config["robot"] = robot
                goal_config["train"] = self.train
                goal_config["normalize_rewards"] = self.normalize_rewards
                goal_config["normalize_observations"] = self.normalize_observations
                goal_config["max_steps"] = self.max_steps_per_episode
                new_goal:Goal = GoalRegistry.get(goal_type)(**goal_config)
                self.goals.append(new_goal)
                robot.set_goal(new_goal)

        # init sensors that don't belong to a robot
        if "sensors" in env_config:
            for sensor_entry in env_config["sensors"]:
                sensor_type = sensor_entry["type"]
                sensor_config = sensor_entry["config"]
                sensor_config["sim_step"] = self.sim_step
                sensor_config["normalize"] = self.normalize_observations
                sensor_config["sim_steps_per_env_step"] = self.sim_steps_per_env_step
                new_sensor:Sensor = SensorRegistry.get(sensor_type)(**sensor_config)
                self.sensors.append(new_sensor)

        # register robots with the world
        self.world.register_robots(self.robots)