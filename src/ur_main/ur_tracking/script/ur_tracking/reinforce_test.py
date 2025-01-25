#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import rospy
import gym
from ur_tracking.env.ur_tracking_env import URSimTracking
from ur_tracking.algorithm.REINFORCEAgent import REINFORCEAgent
from utils import plot_test_results


def run_test(env, agent, episodes=10):
    rewards = []
    success_rates = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        success_times = 0
        total_times = 0
        
        while not done:
            obs = np.array(obs, dtype=np.float32).reshape((1, -1))
            action = agent.control(obs)
            obs, reward, done, info = env.step(action)
            
            if not isinstance(reward, float):
                reward = reward.item()
            total_reward += reward
            
            success, in_test = info["success_rate"]
            if success is True:
                success_times += 1
            total_times += 1

        
        rewards.append(total_reward)
        print(f'Episode {episode + 1}: Total Reward = {total_reward}')
        success_rates.append(round(success_times / total_times, 2))
    return rewards, success_rates

def test():
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)
    ws_path = rospy.get_param('/ws_path')
    env = gym.make('URSimTracking-v0')
    
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed=seed)

    obs_dim, n_act = env.observation_space.shape[0], env.action_space.shape[0]
        
    tester = REINFORCEAgent(obs_dim, n_act, epochs=10, hdim=16, lr=1e-5, seed=seed)
    
    tester.load_model(policy_weights_path='{}/models_reinforce/z01_new/trained_policy_weights.h5'.format(ws_path))
    rewards, success_rates = run_test(env, tester)
    plot_test_results(rewards, mode='reinforce', texts="Success rates are: {}".format(success_rates))

if __name__ == '__main__':
    test()
