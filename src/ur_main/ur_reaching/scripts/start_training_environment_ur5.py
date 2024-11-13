#!/usr/bin/env python3

import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment


if __name__ == '__main__':
    # Init node
    rospy.init_node('ur5_enviro', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/ur5/task_and_robot_environment_name')
    # Create the Gym environment
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.logdebug("Gym environment done")

    rospack = rospkg.RosPack()
    #env = wrappers.Monitor(env, outdir, video_callable=False, force=True)

    nepisodes = rospy.get_param("/ur5/qlearn/nepisodes")
    
    # Starts the main training loop: the one about the episodes to do
    for i_episode in range(nepisodes):        
        observation = env.reset()
        if i_episode >= 2:
            res = env.testGrasp()
            print('xxxxxxxxxxxxxxxxxxxx\n',res, '\nxxxxxxxxxxxxxxxxxxxx')
    env.close()
