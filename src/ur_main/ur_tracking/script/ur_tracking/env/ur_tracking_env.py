#!/usr/bin/env python3
import copy
import numpy as np
import math
import sys
import time

# ROS 
import rospy
import tf
from ur_tracking.env.joint_publisher import JointPub
from ur_tracking.env.joint_traj_publisher import JointTrajPub
from ur_tracking.env.collision_publisher import CollisionPublisher

# Gazebo
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, GetModelState
from gazebo_msgs.srv import GetWorldProperties
from gazebo_msgs.msg import LinkStates

# For reset GAZEBO simultor
from ur_tracking.env.gazebo_connection import GazeboConnection
from ur_tracking.env.controllers_connection import ControllersConnection

# ROS msg
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteModel

# Gym
import gym
from gym import utils, spaces
from gym.utils import seeding
# For register my env
from gym.envs.registration import register

# For inherit RobotGazeboEnv
from ur_tracking.env import robot_gazebo_env_goal

# UR5 Utils
from ur_tracking.env.ur_setups import setups
from ur_tracking.env import ur_utils


rospy.loginfo("register...")
#register the training environment in the gym as an available one
reg = gym.envs.register(
    id='URSimTracking-v0',
    entry_point='ur_tracking.env.ur_tracking_env:URSimTracking', # Its directory associated with importing in other sources like from 'ur_reaching.env.ur_sim_env import *' 
    #timestep_limit=100000,
    )

class URSimTracking(robot_gazebo_env_goal.RobotGazeboEnv):
    def __init__(self):
        rospy.logdebug("Starting URSimTracking Class object...")

        # Init GAZEBO Objects
        self.set_obj_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_state = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        # Subscribe joint state and target pose
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/target_blocks_pose", Point, self.target_point_callback)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_state_callback)
        # rospy.Subscriber("/collision_status", Bool, self.collision_status)

        # For checking reset processing.. since collision checking
        self.reset_precessing = False

        # Gets training parameters from param server
        self.desired_pose = Pose()
        self.running_step = rospy.get_param("/running_step")
        self.max_height = rospy.get_param("/max_height")
        self.min_height = rospy.get_param("/min_height")
        self.observations = rospy.get_param("/observations")
        
        # Tracking distance in z
        self.z_dist = rospy.get_param("z_dist")
        self.height = 0.0

        # Test
        self.success = False
        self.in_test = False
        
        # Joint Velocity limitation
        shp_vel_max = rospy.get_param("/joint_velocity_limits_array/shp_max")
        shp_vel_min = rospy.get_param("/joint_velocity_limits_array/shp_min")
        shl_vel_max = rospy.get_param("/joint_velocity_limits_array/shl_max")
        shl_vel_min = rospy.get_param("/joint_velocity_limits_array/shl_min")
        elb_vel_max = rospy.get_param("/joint_velocity_limits_array/elb_max")
        elb_vel_min = rospy.get_param("/joint_velocity_limits_array/elb_min")
        wr1_vel_max = rospy.get_param("/joint_velocity_limits_array/wr1_max")
        wr1_vel_min = rospy.get_param("/joint_velocity_limits_array/wr1_min")
        wr2_vel_max = rospy.get_param("/joint_velocity_limits_array/wr2_max")
        wr2_vel_min = rospy.get_param("/joint_velocity_limits_array/wr2_min")
        wr3_vel_max = rospy.get_param("/joint_velocity_limits_array/wr3_max")
        wr3_vel_min = rospy.get_param("/joint_velocity_limits_array/wr3_min")
        self.joint_velocty_limits = {"shp_vel_max": shp_vel_max,
                             "shp_vel_min": shp_vel_min,
                             "shl_vel_max": shl_vel_max,
                             "shl_vel_min": shl_vel_min,
                             "elb_vel_max": elb_vel_max,
                             "elb_vel_min": elb_vel_min,
                             "wr1_vel_max": wr1_vel_max,
                             "wr1_vel_min": wr1_vel_min,
                             "wr2_vel_max": wr2_vel_max,
                             "wr2_vel_min": wr2_vel_min,
                             "wr3_vel_max": wr3_vel_max,
                             "wr3_vel_min": wr3_vel_min
                             }

        # Joint limitation
        shp_max = rospy.get_param("/joint_limits_array/shp_max")
        shp_min = rospy.get_param("/joint_limits_array/shp_min")
        shl_max = rospy.get_param("/joint_limits_array/shl_max")
        shl_min = rospy.get_param("/joint_limits_array/shl_min")
        elb_max = rospy.get_param("/joint_limits_array/elb_max")
        elb_min = rospy.get_param("/joint_limits_array/elb_min")
        wr1_max = rospy.get_param("/joint_limits_array/wr1_max")
        wr1_min = rospy.get_param("/joint_limits_array/wr1_min")
        wr2_max = rospy.get_param("/joint_limits_array/wr2_max")
        wr2_min = rospy.get_param("/joint_limits_array/wr2_min")
        wr3_max = rospy.get_param("/joint_limits_array/wr3_max")
        wr3_min = rospy.get_param("/joint_limits_array/wr3_min")
        self.joint_limits = {"shp_max": shp_max,
                             "shp_min": shp_min,
                             "shl_max": shl_max,
                             "shl_min": shl_min,
                             "elb_max": elb_max,
                             "elb_min": elb_min,
                             "wr1_max": wr1_max,
                             "wr1_min": wr1_min,
                             "wr2_max": wr2_max,
                             "wr2_min": wr2_min,
                             "wr3_max": wr3_max,
                             "wr3_min": wr3_min
                             }
        #  Init joint pose
        shp_init_value = rospy.get_param("/init_joint_pose/shp")
        shl_init_value = rospy.get_param("/init_joint_pose/shl")
        elb_init_value = rospy.get_param("/init_joint_pose/elb")
        wr1_init_value = rospy.get_param("/init_joint_pose/wr1")
        wr2_init_value = rospy.get_param("/init_joint_pose/wr2")
        wr3_init_value = rospy.get_param("/init_joint_pose/wr3")
        self.init_joint_pose = [shp_init_value, shl_init_value, elb_init_value, wr1_init_value, wr2_init_value, wr3_init_value]
        self.joint_pose = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)

        # 3D coordinate limits
        x_max = rospy.get_param("/cartesian_limits/x_max")
        x_min = rospy.get_param("/cartesian_limits/x_min")
        y_max = rospy.get_param("/cartesian_limits/y_max")
        y_min = rospy.get_param("/cartesian_limits/y_min")
        z_max = rospy.get_param("/cartesian_limits/z_max")
        z_min = rospy.get_param("/cartesian_limits/z_min")       
        self.xyz_limits = {"x_max": x_max,
                            "x_min": shp_vel_min,
                            "y_max": y_max,
                            "y_min": y_min,
                            "z_max": z_max,
                            "z_min": z_min
                            }

        # Fill in the Done Episode Criteria list
        self.episode_done_criteria = rospy.get_param("/episode_done_criteria")
        
        # stablishes connection with simulator
        self._gz_conn = GazeboConnection()
        self._ctrl_conn = ControllersConnection(namespace="")
        
        # Controller type for ros_control
        self._ctrl_type =  rospy.get_param("/control_type")
        self.pre_ctrl_type =  self._ctrl_type

        # We init the observations
        self.base_orientation = Quaternion()
        self.target_point = Point()
        self.link_state = LinkStates()
        self.joints_state = JointState()
        self.end_effector = Point() 
        self.distance = None
        self.prev_distance = -np.inf
        self.is_tracking = False

        # Arm/Control parameters
        self._ik_params = setups['UR5_6dof']['ik_params']
        
        # ROS msg type
        self._joint_pubisher = JointPub()
        self._joint_traj_pubisher = JointTrajPub()

        # Gazebo collsion msg puiblisher
        # self._collision_publisher = CollisionPublisher()

        # Gym interface and action
        # self.action_space = spaces.Discrete(6)
        self.observation_space = 15 #np.arange(self.get_observations().shape[0])
        self.reward_range = (-np.inf, np.inf)
        self._seed()

        # Change the controller type 
        set_joint_vel_server = rospy.Service('/set_velocity_controller', SetBool, self._set_vel_ctrl)
        set_joint_traj_vel_server = rospy.Service('/set_trajectory_velocity_controller', SetBool, self._set_traj_vel_ctrl)

        self.vel_traj_controller = ['joint_state_controller',
                            'gripper_controller',
                            'vel_traj_controller']
        self.vel_controller = ["joint_state_controller",
                                "gripper_controller",
                                "ur_shoulder_pan_vel_controller",
                                "ur_shoulder_lift_vel_controller",
                                "ur_elbow_vel_controller",
                                "ur_wrist_1_vel_controller",
                                "ur_wrist_2_vel_controller",
                                "ur_wrist_3_vel_controller"]
        # Helpful False
        self.stop_flag = False
        stop_trainning_server = rospy.Service('/stop_training', SetBool, self._stop_trainnig)
        start_trainning_server = rospy.Service('/start_training', SetBool, self._start_trainnig)
        self._ctrl_conn.load_controllers("joint_state_controller")
        
        self.set_act_obs_space()

    def set_act_obs_space(self):
        self.obs_space_low = np.array(
            [self.joint_limits["shp_min"], self.joint_limits["shl_min"], self.joint_limits["elb_min"], self.joint_limits["wr1_min"], \
            # self.joint_limits["wr2_min"], self.joint_limits["wr3_min"], \
            self.joint_velocty_limits["shp_vel_min"], self.joint_velocty_limits["shl_vel_min"], self.joint_velocty_limits["elb_vel_min"], self.joint_velocty_limits["wr1_vel_min"], \
            # self.joint_velocty_limits["wr2_vel_min"], self.joint_velocty_limits["wr3_vel_min"], \
            self.xyz_limits["x_min"],  self.xyz_limits["y_min"],  self.xyz_limits["z_min"], -1.0], dtype=np.float32) # -1.0 for shape of adding target_x
        self.obs_space_high = np.array(
            [self.joint_limits["shp_max"], self.joint_limits["shl_max"], self.joint_limits["elb_max"], self.joint_limits["wr1_max"], \
            # self.joint_limits["wr2_max"], self.joint_limits["wr3_max"], \
            self.joint_velocty_limits["shp_vel_max"], self.joint_velocty_limits["shl_vel_max"], self.joint_velocty_limits["elb_vel_max"], self.joint_velocty_limits["wr1_vel_max"], \
            # self.joint_velocty_limits["wr2_vel_max"],  self.joint_velocty_limits["wr3_vel_max"],  \
            self.xyz_limits["x_max"],  self.xyz_limits["y_max"],  self.xyz_limits["z_max"], 1.0], dtype=np.float32) # 1.0 for shape of adding target_x
        observation_space = spaces.Box(
            low=self.obs_space_low, \
            high=self.obs_space_high, \
            dtype=np.float32)
        self.observation_space = observation_space
        action_space = spaces.Box(
            low=np.array(
                [self.joint_velocty_limits["shp_vel_min"], self.joint_velocty_limits["shl_vel_min"], self.joint_velocty_limits["elb_vel_min"], self.joint_velocty_limits["wr1_vel_min"], \
                #  self.joint_velocty_limits["wr2_vel_min"],  self.joint_velocty_limits["wr3_vel_min"] \
                ], dtype=np.float32), \
            high=np.array([
                self.joint_velocty_limits["shp_vel_max"], self.joint_velocty_limits["shl_vel_max"], self.joint_velocty_limits["elb_vel_max"], self.joint_velocty_limits["wr1_vel_max"], \
                #  self.joint_velocty_limits["wr2_vel_max"],  self.joint_velocty_limits["wr3_vel_max"] \
                ], dtype=np.float32), \
                dtype=np.float32)
        self.action_space = action_space
    
    def check_stop_flg(self):
        if self.stop_flag is False:
            return False
        else:
            return True

    def _start_trainnig(self, req):
        rospy.logdebug("_start_trainnig!!!!")
        self.stop_flag = False
        return SetBoolResponse(True, "_start_trainnig")

    def _stop_trainnig(self, req):
        rospy.logdebug("_stop_trainnig!!!!")
        self.stop_flag = True
        return SetBoolResponse(True, "_stop_trainnig")

    def _set_vel_ctrl(self, req):
        rospy.wait_for_service('set_velocity_controller')
        self._ctrl_conn.stop_controllers(self.vel_traj_controller)
        self._ctrl_conn.start_controllers(self.vel_controller)
        self._ctrl_type = 'vel'
        return SetBoolResponse(True, "_set_vel_ctrl")

    def _set_traj_vel_ctrl(self, req):
        rospy.wait_for_service('set_trajectory_velocity_controller')
        self._ctrl_conn.stop_controllers(self.vel_controller)
        self._ctrl_conn.start_controllers(self.vel_traj_controller)    
        self._ctrl_type = 'traj_vel'
        return SetBoolResponse(True, "_set_traj_vel_ctrl")  

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def link_state_callback(self, msg):
        self.link_state = msg
        # self.end_effector = self.link_state.pose[8]
        try:
            idx_left = msg.name.index("robot::robotiq_85_left_finger_tip_link")
            idx_right = msg.name.index("robot::robotiq_85_right_finger_tip_link")
            left_tip, right_tip = msg.pose[idx_left], msg.pose[idx_right]

            self.end_effector.x = (left_tip.position.x + right_tip.position.x) / 2.0
            self.end_effector.y = (left_tip.position.y + right_tip.position.y) / 2.0
            self.end_effector.z = (left_tip.position.z + right_tip.position.z) / 2.0
            # self.end_effector.orientation = left_tip.orientation
        except ValueError:
            rospy.logwarn("One or both tip links not found in /gazebo/link_states.")
            
    def target_point_callback(self, msg):
        self.target_point = msg

    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                # # if reset world
                # self._ctrl_conn.start_controllers(controllers_on="joint_state_controller")
                # if reset sim
                self._ctrl_conn.stop_all_controller()
                for ctrl in self.vel_controller: 
                    self._ctrl_conn.unload_controllers(ctrl)
                    self._ctrl_conn.load_controllers(ctrl)
                self._ctrl_conn.reset_joint_controllers("vel")
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))
        
        target_pose_msg = None
        while target_pose_msg is None and not rospy.is_shutdown():
            try:
                target_pose_msg = rospy.wait_for_message("/target_blocks_pose", Point, timeout=0.1)
                self.target_point = target_pose_msg
                rospy.logdebug("Reading target pose READY")
            except Exception as e:
                rospy.logdebug("Reading target pose not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")

    def collision_status(self, msg):
        
        if msg.data == True and self.reset_precessing == False:
            self.reset()
            print("###### Colliding ! #####")

    def get_xyz(self, q):
        """Get x,y,z coordinates 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz

    def get_current_xyz(self):
        return
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[0]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[2]
        wr1_joint_ang = joint_states.position[3]
        wr2_joint_ang = joint_states.position[4]
        wr3_joint_ang = joint_states.position[5]
        
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz
            
    def get_orientation(self, q):
        """Get Euler angles 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        orientation = mat[0:3, 0:3]
        roll = -orientation[1, 2]
        pitch = orientation[0, 2]
        yaw = -orientation[0, 1]
        
        return Vector3(roll, pitch, yaw)

    # def cvt_quat_to_euler(self, quat):
    #     euler_rpy = Vector3()
    #     euler = tf.transformations.euler_from_quaternion(
    #         [self.quat.x, self.quat.y, self.quat.z, self.quat.w])

    #     euler_rpy.x = euler[0]
    #     euler_rpy.y = euler[1]
    #     euler_rpy.z = euler[2]
    #     return euler_rpy

    def init_joints_pose(self, init_pose):
        """
        We initialise the Position variable that saves the desired position where we want our
        joints to be
        :param init_pos:
        :return:
        """

        # self.current_joint_pose =[]
        # self.current_joint_pose = copy.deepcopy(init_pos)
        # return self.current_joint_pose
        count = 0
        tolerance = 0.01
        # max_velocity = 0.5

        current_pose = self.joint_pose
        errors = np.abs(np.array(init_pose) - np.array(current_pose))

        while np.any(errors > tolerance):
            
            joint_vels = [
                errors[i] if init_pose[i] - current_pose[i] > 0 else -errors[i]
                for i in range(len(init_pose))
            ]
            self._joint_pubisher.move_joints(joint_vels)
            rospy.sleep(0.1)

            current_pose = self.joint_pose
            errors = np.abs(np.array(init_pose) - np.array(current_pose))

            if count >= 100:
                rospy.sleep(1)
                rospy.signal_shutdown("Closing RobotGazeboEnvironment")
            count += 1
            print("init {} times...".format(count))

        stop_joint_vels = [0.0] * len(init_pose)
        self._joint_pubisher.move_joints(stop_joint_vels)

    def get_euclidean_dist(self, p_in, p_pout):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((p_in.x, p_in.y, p_in.z))
        b = np.array((p_pout.x, p_pout.y, p_pout.z))

        distance = np.linalg.norm(a - b)

        return distance

    def joints_state_callback(self, msg):
        self.joints_state = msg
        self.joint_pose = np.array([self.joints_state.position[3], self.joints_state.position[2], self.joints_state.position[0],
                                    self.joints_state.position[4], self.joints_state.position[5], self.joints_state.position[6]], dtype=np.float32)

    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array
        :return: observation
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[3]
        shl_joint_ang = joint_states.position[2]
        elb_joint_ang = joint_states.position[0]
        wr1_joint_ang = joint_states.position[4]
        wr2_joint_ang = joint_states.position[5]
        wr3_joint_ang = joint_states.position[6]

        shp_joint_vel = joint_states.velocity[3]
        shl_joint_vel = joint_states.velocity[2]
        elb_joint_vel = joint_states.velocity[0]
        wr1_joint_vel = joint_states.velocity[4]
        wr2_joint_vel = joint_states.velocity[5]
        wr3_joint_vel = joint_states.velocity[6]

        # q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        # eef_x, eef_y, eef_z = self.get_xyz(q)

        observation = []
        rospy.logdebug("List of Observations==>"+str(self.observations))
        for obs_name in self.observations:
            if obs_name == "shp_joint_ang":
                observation.append(shp_joint_ang)
            elif obs_name == "shl_joint_ang":
                observation.append(shl_joint_ang)
            elif obs_name == "elb_joint_ang":
                observation.append(elb_joint_ang)
            elif obs_name == "wr1_joint_ang":
                observation.append(wr1_joint_ang)
            elif obs_name == "wr2_joint_ang":
                observation.append(wr2_joint_ang)
            elif obs_name == "wr3_joint_ang":
                observation.append(wr3_joint_ang)
            elif obs_name == "shp_joint_vel":
                observation.append(shp_joint_vel)
            elif obs_name == "shl_joint_vel":
                observation.append(shl_joint_vel)
            elif obs_name == "elb_joint_vel":
                observation.append(elb_joint_vel)
            elif obs_name == "wr1_joint_vel":
                observation.append(wr1_joint_vel)
            elif obs_name == "wr2_joint_vel":
                observation.append(wr2_joint_vel)
            elif obs_name == "wr3_joint_vel":
                observation.append(wr3_joint_vel)
            elif obs_name == "eef_x":
                # observation.append(eef_x)
                observation.append(self.end_effector.x)
            elif obs_name == "eef_y":
                # observation.append(eef_y)
                observation.append(self.end_effector.y)
            elif obs_name == "eef_z":
                # observation.append(eef_z)
                observation.append(self.end_effector.z)
            elif obs_name == "target_x":
                observation.append(self.target_point.x)
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

        return observation

    def clamp_to_joint_limits(self):
        """
        clamps self.current_joint_pose based on the joint limits
        self._joint_limits
        {
         "shp_max": shp_max,
         "shp_min": shp_min,
         ...
         }
        :return:
        """

        rospy.logdebug("Clamping current_joint_pose>>>" + str(self.current_joint_pose))
        shp_joint_value = self.current_joint_pose[0]
        shl_joint_value = self.current_joint_pose[1]
        elb_joint_value = self.current_joint_pose[2]
        wr1_joint_value = self.current_joint_pose[3]
        wr2_joint_value = self.current_joint_pose[4]
        wr3_joint_value = self.current_joint_pose[5]

        self.current_joint_pose[0] = max(min(shp_joint_value, self._joint_limits["shp_max"]),
                                         self._joint_limits["shp_min"])
        self.current_joint_pose[1] = max(min(shl_joint_value, self._joint_limits["shl_max"]),
                                         self._joint_limits["shl_min"])
        self.current_joint_pose[2] = max(min(elb_joint_value, self._joint_limits["elb_max"]),
                                         self._joint_limits["elb_min"])
        self.current_joint_pose[3] = max(min(wr1_joint_value, self._joint_limits["wr1_max"]),
                                         self._joint_limits["wr1_min"])
        self.current_joint_pose[4] = max(min(wr2_joint_value, self._joint_limits["wr2_max"]),
                                         self._joint_limits["wr2_min"])
        self.current_joint_pose[5] = max(min(wr3_joint_value, self._joint_limits["wr3_max"]),
                                         self._joint_limits["wr3_min"])

        rospy.logdebug("DONE Clamping current_joint_pose>>>" + str(self.current_joint_pose))

    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        # 0st: We pause the Simulator
        rospy.logdebug("Pausing SIM...")
        self.reset_precessing = True
        self._gz_conn.pauseSim()

        # 1st: resets the simulation to initial values
        rospy.logdebug("Reset SIM...")
        self._gz_conn.resetWorld()
        # self._gz_conn.resetSim()

        # 2nd: We Set the gravity to 0.0 so that we dont fall when reseting joints
        # It also UNPAUSES the simulation
        rospy.logdebug("Remove Gravity...")
        self._gz_conn.change_gravity_zero()

        # EXTRA: Reset JoinStateControlers because sim reset doesnt reset TFs, generating time problems
        # rospy.logdebug("reset_ur_joint_controllers...")
        # self._ctrl_conn.reset_ur_joint_controllers(self._ctrl_type)
        self._ctrl_type = "vel"
        # self._ctrl_conn.reset_joint_controllers(self._ctrl_type)
        rospy.logdebug("check_all_systems_ready...")
        '''Here controllers are reloaded'''
        self.check_all_systems_ready()
        
        # 3rd: resets the robot to initial conditions
        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose))
        # We save that position as the current joint desired position
        self.init_joints_pose(self.init_joint_pose)
        rospy.logwarn("UR pose initialized!!!")
        
        #####
        rate = rospy.Rate(10)  # 10Hz 循环频率
        while not rospy.is_shutdown() and self.target_point.x >= 1.19:
            rospy.loginfo(f"x = {self.target_point.x}, waiting...")
            rate.sleep()  # 每次循环时休眠

        # 4th: We Set the init pose to the jump topic so that the jump control can update
        # We check the jump publisher has connection

        if self._ctrl_type == 'traj_vel':
            self._joint_traj_pubisher.check_publishers_connection()
        elif self._ctrl_type == 'vel':
            self._joint_pubisher.check_publishers_connection()
        else:
            rospy.logwarn("Controller type is wrong!!!!")
        
        # 5th: Check all subscribers work.
        # Get the state of the Robot defined by its RPY orientation, distance from
        # desired point, contact force and JointState of the three joints

        # 6th: We restore the gravity to original
        rospy.logdebug("Restore Gravity...")
        # self._gz_conn.adjust_gravity()

        # 7th: pauses simulation
        rospy.logdebug("Pause SIM...")
        self._gz_conn.pauseSim()
        # self._init_obj_pose()

        self.success = False
        self.in_test = False

        # 8th: Get the State Discrete Stringuified version of the observations
        rospy.logdebug("get_observations...")
        observation = self.get_observations()
        
        self.reset_precessing = True
        return observation
       
    def step(self, action):
        '''
        ('action: ', array([ 0.,  0. , -0., -0., -0. , 0. ], dtype=float32))        
        '''
        rospy.logdebug("UR step func")

        self.training_ok()

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        # Act
        self._gz_conn.unpauseSim()
        # action = action * 0.1
        # print("Act: ", action)
        self._act(action)
        
        # Then we send the command to the robot and let it go
        # for running_step seconds
        rospy.sleep(self.running_step)
        # self._joint_pubisher.move_joints([0.0]*6)
        # rospy.sleep(self.running_step)
        self._gz_conn.pauseSim()

        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.get_observations()

        # finally we get an evaluation based on what happened in the sim
        reward, done = self.step_reward_done()

        # For test
        # if -0.5 <= self.target_point.x <= 0.5:
        #     self.in_test = True
        #     if self.distance <= 0.1:
        #         self.success = True
        if self.distance <= 0.1:
                self.success = True

        return observation, reward, done, {"success_rate": (self.success, self.in_test)} # info
    
    def _act(self, action):
        action = list(action)
        if len(action) == 4:
            action.extend([0.0, 0.0])
        
        self._joint_pubisher.move_joints(action)
        
    def training_ok(self):
        rate = rospy.Rate(1)
        while self.check_stop_flg() is True:                  
            rospy.logdebug("stop_flag is ON!!!!")
            self._gz_conn.unpauseSim()

            if self.check_stop_flg() is False:
                break 
            rate.sleep()
        
    def compute_dist_rewards(self):
        end_effector_pose = np.array([self.end_effector.x, self.end_effector.y, self.end_effector.z])
        self.distance = np.linalg.norm(end_effector_pose - [self.target_point.x, self.target_point.y, self.target_point.z+self.z_dist], axis=0)
        
        self.height = self.end_effector.z - self.target_point.z
        tolerance = 0.1
        if self.height < 0.01:
            rospy.logwarn("xxx End effector is too low xxx")
            reward = -100.0
        # elif 0.01 <= self.height < self.z_dist-tolerance: #self.z_dist
        #     rospy.logwarn("***End effector is too low***")
        #     reward = -1.0
        else:
            progress = self.prev_distance - self.distance
            if self.distance < tolerance:
                rospy.logerr("***End effector is tracking target now***")
                self.is_tracking = True
                if progress > 0.0:
                    reward = 10.0
                else:
                    reward = -np.exp(-self.distance) #-np.exp(self.distance)/2 # 
            else:
                self.is_tracking = False
                if progress > 0.0:
                    reward = np.exp(-self.distance) #np.exp(-self.distance) # 
                else:
                    if 0.01 <= self.height < 0.05: #self.z_dist-tolerance:
                        reward = -10.0
                    else:
                        reward = -np.exp(-self.distance) #-np.exp(self.distance)/2 #
    
        self.prev_distance = copy.deepcopy(self.distance)
        print("-----------------------------------------")
        rospy.loginfo("Height: " + str(self.height)[:4] + "->Dist: " + str(self.distance)[:4] + "->Reward: " + str(reward)[:4])
        print("-----------------------------------------")
        return reward
    
    def step_reward_done(self):
        reward = self.compute_dist_rewards()
        done = False

        if reward == -100.0:
            done = True
        
        if self.target_point.x < -0.79:
            done = True

        if self.target_point.y != -0.8 or self.target_point.z != 0.2:
            reward = -100.0
            done = True
        
        return reward, done
