#!/usr/bin/env python3
# max_iterations: 40
import os
import math
import numpy as np
import rospy
import tf
from openai_ros.robot_envs import ur5_env
from gym import spaces
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher

class UR5PositionEnv(ur5_env.UR5Env):

    """ Task enviroment to train Reinforcement Learning algorithms
    on the UR5 robot using ROS. """

    def __init__(self):

        ros_ws_abspath = rospy.get_param("/ur5/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath \
         in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), \
        "The Simulation ROS Workspace path " + ros_ws_abspath + \
        " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
        "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Start ROS launch that creates the world where the robot lives
        ROSLauncher(rospackage_name="my_ur5_description",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load params from YAML file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/ur5/config",
                               yaml_file_name="ur5_position_task.yaml")

        # Get parameters from YAML file
        self.get_params()
        # Define action space
        self.action_space = spaces.Discrete(self.n_actions)
        # Define observation space
        high = np.array([np.pi, np.pi, np.pi])
        low = np.array([-np.pi, -np.pi, -np.pi])
        self.observation_space = spaces.Box(low, high)
        
        '''pose mode'''
        # # current score for reward estimating
        # self.goal_array = np.array(list(self.goal_pos.values()))
        # self.prev_pos_dist = self.init_distance(self.init_pos) # init_pose from get_params
        '''position mode'''
        self.listener = tf.TransformListener()
        self.init_coord_dist = None
        self.prev_coord_dist = None
        self.curr_coord = None
        self.diff = np.inf
        self.goaled = 0
        self.switch = False

        self.non_movement = 0

        # Add init functions prior to starting the enviroment
        super(UR5PositionEnv, self).__init__(ros_ws_abspath=ros_ws_abspath)


    def get_params(self):

        """ Gets configuration parameters from YAML file.
        Additionally creates a new parameter to track the current
        iteration the robot is at within an episode.
        :return:
        """

        self.n_actions = rospy.get_param('/ur5/n_actions')
        self.n_observations = rospy.get_param('/ur5/n_observations')
        self.max_iterations = rospy.get_param('/ur5/max_iterations')
        self.init_pos = rospy.get_param('/ur5/init_pos')
        self.goal_pos = rospy.get_param('/ur5/goal_pos')
        self.pos_step = rospy.get_param('/ur5/position_delta')
        self.non_movement_penalty = rospy.get_param('/ur5/non_movement_penalty')
        self.reached_goal_reward = rospy.get_param('/ur5/reached_goal_reward')
        self.check_move = rospy.get_param('/ur5/check_move')
        self.non_movement_done = rospy.get_param('/ur5/non_movement_done')
        self.goal_coord = rospy.get_param('/ur5/goal_coord')
        self.current_iteration = 0
    
    # def init_distance(self, init_array):
    #     init_array = np.array(list(self.init_pos.values()))
    #     euclid_dist = np.sqrt(np.sum((init_array - self.goal_array)**2))
    #     return euclid_dist

    def _init_coord_dist(self):
        begin = self.get_gripper_coord()
        print("INIT: ", begin)
        self.init_coord_dist = np.sqrt(np.sum((begin - np.asarray(self.goal_coord))**2))
        self.prev_coord_dist = np.copy(self.init_coord_dist)
    
    def get_gripper_coord(self): #'ee_link'
        self.listener.waitForTransform('/base_link', '/left_inner_finger_pad', rospy.Time(), rospy.Duration(3.0))  # '/wrist_3_link'
        try:
            (trans, rot) = self.listener.lookupTransform('/base_link', '/left_inner_finger_pad', rospy.Time(0))  # '/wrist_3_link'
            #rospy.sleep(0.2)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            raise RuntimeError("Cannot find endeffector coord")
        return np.asarray(trans)
    
    def _set_action(self, action):

        """ Maps action identifiers into actual
         movements of the robot.
         :return:
         """
        # Here added return term
        if action == 0:
            rospy.loginfo("SHOULDER PAN => +")
            self.pos['shoulder_pan_joint'] += self.pos_step
            return self.move_shoulder_pan_joint(self.pos)
        elif action == 1: 
            rospy.loginfo("SHOULDER PAN => -")
            self.pos['shoulder_pan_joint'] -= self.pos_step
            return self.move_shoulder_pan_joint(self.pos)

        elif action == 2:
            rospy.loginfo("SHOULDER LIFT => +")
            self.pos['shoulder_lift_joint'] += self.pos_step
            return self.move_shoulder_lift_joint(self.pos)
        elif action == 3:
            rospy.loginfo("SHOULDER LIFT => -")
            self.pos['shoulder_lift_joint'] -= self.pos_step
            return self.move_shoulder_lift_joint(self.pos)

        elif action == 4:
            rospy.loginfo("ELBOW => +")
            self.pos['elbow_joint'] += self.pos_step
            return self.move_elbow_joint(self.pos)
        elif action == 5:
            rospy.loginfo("ELBOW => -")
            self.pos['elbow_joint'] -= self.pos_step
            return self.move_elbow_joint(self.pos)
        
        elif action == 6:
            rospy.loginfo("WIRST 1 => +")
            self.pos['wrist_1_joint'] += self.pos_step
            return self.move_wrist_1_joint(self.pos)
        elif action == 7:
            rospy.loginfo("WIRST 1 => -")
            self.pos['wrist_1_joint'] -= self.pos_step
            return self.move_wrist_1_joint(self.pos)
        
        elif action == 8:
            rospy.loginfo("WIRST 2 => +")
            self.pos['wrist_2_joint'] += self.pos_step
            return self.move_wrist_2_joint(self.pos)
        elif action == 9:
            rospy.loginfo("WIRST 2 => -")
            self.pos['wrist_2_joint'] -= self.pos_step
            return self.move_wrist_2_joint(self.pos)
        
        elif action == 10:
            rospy.loginfo("WIRST 3 => +")
            self.pos['wrist_3_joint'] += self.pos_step
            return self.move_wrist_3_joint(self.pos)
        elif action == 11:
            rospy.loginfo("WIRST 3 => -")
            self.pos['wrist_3_joint'] -= self.pos_step
            return self.move_wrist_3_joint(self.pos)
    
    def _set_gripper_status(self, status):
        if status == 0:
            rospy.loginfo("GRIPPPER => open")
            #self.pos['finger_joint'] = 0.0
            return self.move_finger_joint({'finger_joint': 0.0})
        elif status == 1:
            rospy.loginfo("GRIPPPER => close")
            #self.pos['finger_joint'] = 0.8
            return self.move_finger_joint({'finger_joint': 0.8})
        else:
            return None

    def _get_obs(self):

        """ Stores the current position of the three moving joints
        in a numpy array.
        :return:obs
        """

        obs = np.array([self.joints['shoulder_pan_joint'],
                        self.joints['shoulder_lift_joint'],
                        self.joints['elbow_joint'],
                        self.joints['wrist_1_joint'],
                        self.joints['wrist_2_joint'],
                        self.joints['wrist_3_joint']
                        ])

        return obs

    def _is_done(self, observations, moved):

        """" The episode is done when the robot achieves the desired pose
        with an absolute tolerance of 0.2 per joint.
        :return:done
        """

        done = False
        tolerance = 0.1 # Original 0.2
        
        if not moved and self.check_move and self.non_movement_done:
            if self.non_movement >= 2:
                done = True
                self.current_iteration = 0
                self.non_movement = 0
                return done
            else: self.non_movement += 1
        elif moved: self.non_movement = 0

        self.curr_coord = self.get_gripper_coord()
        self.diff = np.sqrt(np.sum((self.curr_coord - self.goal_coord)**2))
        print("Current distance:", self.diff)
        if self.diff < 0.05:
            done = True
            rospy.logerr("Goal coordinate reached !!!")
        elif self.current_iteration == (self.max_iterations-1):
            # Return done at the end of an episode
            done = True
            self.current_iteration = 0 # last step in episode, reset, to mark for _compute_reward
        # print(self.curr_coord)
        return done

    def _compute_reward(self, observations, done, info, moved):

        """
        Gives more points for staying closer to the goal position.
        A fixed reward of 100 is given when the robot achieves the
        goal position.
        :return:reward
        """
        # print(self.current_iteration)
        if not moved and self.check_move:
            reward = self.non_movement_penalty
            self.current_iteration += 1
            return reward
        
        curr_coord_dist = np.sqrt(np.sum((self.curr_coord - self.goal_coord)**2))
        score = 1 / curr_coord_dist
        progress = self.prev_coord_dist - curr_coord_dist
        if not done:
            if progress > 0.0:
                if info and not self.switch:
                    reward = 50
                    self.switch = True
                else:
                    reward = score ## use progress as reward
            else:
                reward = -score
                # if info:
                #     reward = -curr_coord_dist * 100
                # else:
                #     reward = -score #-curr_coord_dist
            self.prev_coord_dist = np.copy(curr_coord_dist)
            self.current_iteration += 1
        elif done and self.current_iteration == 0: # last but not reached
            # If done due to be the last step of episode (we just reseted it in _is_done())
            # then compute reward
            # if progress > 0.0:
            #     reward = score
            # else:
            #     reward = -curr_coord_dist * 10
            reward = score
            self.prev_coord_dist = np.copy(self.init_coord_dist)
            self.diff = np.inf
            self.switch = False
        elif done and self.current_iteration > 0:
            # If done at an iteration greater than 0, then it must has reached
            reward = self.reached_goal_reward
            self.goaled += 1
            self.current_iteration = 0
            self.curr_coord = None
            self.prev_coord_dist = np.copy(self.init_coord_dist)
            self.diff = np.inf
            self.switch = False
        else:
            rospy.logdebug("Unknown goal status => Reward?")
            reward = 0
            self.current_iteration = 0
            self.curr_coord = None
            self.prev_coord_dist = np.copy(self.init_coord_dist)
            self.diff = np.inf
            self.switch = False
        
        rospy.logwarn("Goal reached " + str(self.goaled) + " times.")
        return reward

    def _get_info(self):
        ### Check if the coarse positioning is complete, and if so, begin fine-positioning.
        if self.diff <= 0.1: #0.12: # 0.15
            return True
        return False
    
    def _init_env_variables(self):

        """
        This variable needs to be implemented but
        it's not used, hence it just passes.
        :return:
        """

        pass

    def _set_init_pose(self):

        """
        Sets joints to initial position [0,0,0,0,0,0]
        :return:
        """

        rospy.logdebug('Checking publishers connection')
        self.check_publishers_connection()

        rospy.logdebug('Reseting to initial robot position')
        # Reset internal position variable
        self.init_internal_vars(self.init_pos)
        # Move joints to origin
        _ = self.move_shoulder_pan_joint(self.init_pos)
        _ = self.move_shoulder_lift_joint(self.init_pos)
        _ = self.move_elbow_joint(self.init_pos)
        _ = self.move_wrist_1_joint(self.init_pos)
        _ = self.move_wrist_2_joint(self.init_pos)
        _ = self.move_wrist_3_joint(self.init_pos)
        _ = self.move_finger_joint({'finger_joint': 0.8})
