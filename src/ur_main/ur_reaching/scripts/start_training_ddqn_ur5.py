#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from gym import wrappers
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from ddqn_constructor import create_agent
from utils import plot_results, obs_to_state


if __name__ == '__main__':
    ## Init ROS node and create OpenAI_ROS Env
    rospy.init_node('ur5_ddqn', anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = rospy.get_param(
        '/ur5/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.logwarn("Gym environment done")

    ## set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ur_reaching')
    outdir_monitor = pkg_path + '/training_results/monitor_ddqn'
    env = wrappers.Monitor(env, outdir_monitor, video_callable=False, force=True)
    
    ## load parameters
    gamma = rospy.get_param("/ur5/ddqn/gamma")
    batch_size = rospy.get_param("ur5/ddqn/batch_size")
    capacity = rospy.get_param("ur5/ddqn/capacity")
    lr = rospy.get_param("/ur5/ddqn/lr")
    nepisodes = rospy.get_param("/ur5/ddqn/nepisodes")
    nsteps = rospy.get_param("/ur5/ddqn/nsteps")
    round_format = rospy.get_param("/ur5/ddqn/round_format")
    num_states = rospy.get_param("/ur5/ddqn/num_states")
    num_mid = rospy.get_param("/ur5/ddqn/num_mid")
    num_actions = env.action_space.n
    is_training = True

    '''Training process'''
    agent = create_agent(num_states, num_mid, num_actions,
                         gamma=gamma, batch_size=batch_size,
                         capacity=capacity, lr=lr)

    highest_reward = 0
    rospy.logwarn("Starting to train the robot...")

    for i_episode in range(nepisodes):

        observation = env.reset()
        done = False
        cumulated_reward = 0
        state = obs_to_state(observation, round_format)

        ## train robot nsteps to reach the goal
        for i in range(nsteps):
            action = agent.getAction(state, i_episode)
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            next_state = obs_to_state(observation, round_format)

            ## show out training infomations
            print("Episode: {} --> Step: {}".format(i_episode, i))
            rospy.logwarn("# state we were =>" + str(state))
            rospy.logwarn("# action that we took =>" + str(action))
            rospy.logwarn("# reward that action gave =>" + str(reward))
            rospy.logwarn("# episode cumulated_reward =>" + str(cumulated_reward))
            rospy.logwarn("# State for next step =>" + str(next_state))

            if not done:
                agent.memorize(state, action, next_state, reward)
                agent.update_q_function()
                state = np.copy(next_state)
            else:
                break
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

    env.close()
    plot_results(mode=None)