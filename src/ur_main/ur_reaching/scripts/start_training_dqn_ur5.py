#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from gym import wrappers
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from dqn_constructor import create_agent
from utils import plot_results, obs_to_state


if __name__ == '__main__':
    ## Init ROS node and create OpenAI_ROS Env
    rospy.init_node('ur5_dqn', anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = rospy.get_param(
        '/ur5/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.logwarn("Gym environment done")

    ## set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ur_reaching')
    outdir_monitor = pkg_path + '/training_results/monitor_dqn'
    env = wrappers.Monitor(env, outdir_monitor, video_callable=False, force=True)
    
    ## load parameters
    gamma = rospy.get_param("/ur5/dqn/gamma")
    epsilon = rospy.get_param("/ur5/dqn/epsilon")
    ed = rospy.get_param("/ur5/dqn/ed")
    eb = rospy.get_param("/ur5/dqn/eb")
    lr = rospy.get_param("/ur5/dqn/lr")
    nepisodes = rospy.get_param("/ur5/dqn/nepisodes")
    nsteps = rospy.get_param("/ur5/dqn/nsteps")
    round_format = rospy.get_param("/ur5/dqn/round_format")
    num_states = rospy.get_param("/ur5/dqn/num_states")
    num_mid = rospy.get_param("/ur5/dqn/num_mid")
    num_actions = rospy.get_param("/ur5/dqn/num_actions")  # env.action_space.n
    is_training = True

    '''Training process'''
    agent = create_agent(num_states, num_mid, num_actions,
                         gamma=gamma, epsilon=epsilon, lr=lr)

    highest_reward = 0
    rospy.logwarn("Starting to train the robot...")

    for i_episode in range(nepisodes):

        observation = env.reset()
        done = False
        info = False
        cumulated_reward = 0
        state = obs_to_state(observation, round_format)
        
        if is_training and agent.brain.epsilon > eb: # exploration decay
            agent.brain.epsilon *= ed
        
        ## train robot nsteps to reach the goal
        for i in range(nsteps):
            action = agent.getAction(state, is_training, info)
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            next_state = obs_to_state(observation, round_format)

            ## show out training infomations
            rospy.logwarn("# current episode:" + str(i_episode) + " step =>" + str(i))
            rospy.logwarn("# chosen action =>" + str(action))
            rospy.logwarn("# reward of the action =>" + str(reward)[:5])
            rospy.logwarn("# episode cumulated_reward =>" + str(cumulated_reward)[:5])
            rospy.logwarn("# next state =>" + str(next_state))
            
            if is_training:
                agent.updateQnet(state, action, reward, next_state)
            if not done:
                state = np.copy(next_state)
            else:
                break
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

    env.close()
    agent.saveQnet(saving=False)
    plot_results(mode='dqn')

''' some ideas'''
# random initial pose
# random delta