#!/usr/bin/env python3

import numpy
import time
import qlearn
from gym import wrappers
# from functools import reduce
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
# results
from utils import plot_results, save_Q_table, obs_to_state


if __name__ == '__main__':

    rospy.init_node('ur5_qlearn', anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = rospy.get_param(
        '/ur5/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.logdebug("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ur_reaching')
    outdir = pkg_path + '/training_results/monitor_qlearn'
    env = wrappers.Monitor(env, outdir, video_callable=False, force=True)
    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    Alpha = rospy.get_param("/ur5/qlearn/alpha")
    Epsilon = rospy.get_param("/ur5/qlearn/epsilon")
    Gamma = rospy.get_param("/ur5/qlearn/gamma")
    epsilon_discount = rospy.get_param("/ur5/qlearn/epsilon_discount")
    nepisodes = rospy.get_param("/ur5/qlearn/nepisodes")
    nsteps = rospy.get_param("/ur5/qlearn/nsteps")
    round_format = rospy.get_param("/ur5/qlearn/round_format")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon
    start_time = time.time()
    highest_reward = 0
    rospy.logwarn("Starting to train the robot...")

    for i_episode in range(nepisodes):

        cumulated_reward = 0
        done = False
        info, readyToGrasp = False, False
        if qlearn.epsilon > 0.05: # # Original 0.05
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        stateAsList = obs_to_state(observation, round_format)
        state = ''.join(map(str, stateAsList))

        # For each episode, we test the robot for nsteps
        for i in range(nsteps):
            action = qlearn.chooseAction(state, info) # Pick an action based on the current state
            observation, reward, done, info = env.step(action)# Execute action in env and get feedback

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextStateAsList = obs_to_state(observation, round_format)
            nextState = ''.join(map(str, nextStateAsList))

            rospy.logwarn("# current episode:" + str(i_episode) + " step =>" + str(i))
            rospy.logwarn("# chosen action =>" + str(action))
            rospy.logwarn("# reward of the action =>" + str(reward)[:5])
            rospy.logwarn("# episode cumulated_reward =>" + str(cumulated_reward)[:5])
            rospy.logwarn("# next state =>" + str(nextStateAsList))
            qlearn.learn(state, action, reward, nextState)
            
            if not (done): # env feedback done==True
                state = nextState
            elif (done) and readyToGrasp:
                # Greifer 0.1m senken, schliesst, in anderen Ort legen
                # da Greifer liegt oben 0.1m und der Toleranz ist fast gleich Size wie Obj
                # so ist greifen moeglich. 
                pass
            else: break

    '''program end'''
    save_Q_table(Q_values=qlearn.q)
    env.close()
    print("xxxxxxxxxxxxxxxxxxxxx\nEND start_training_qlearn_ur5\nxxxxxxxxxxxxxxxxxxxxx")
    plot_results(mode='qlearn')
