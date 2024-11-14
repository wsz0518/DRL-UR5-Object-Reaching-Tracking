#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import rospy
import gym
from ur_tracking.env.ur_tracking_env import URSimTracking
from ur_tracking.algorithm.PPOGAEAgent import PPOGAEAgent


def run_test(env, agent, episodes=10):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs = np.array(obs, dtype=np.float32).reshape((1, -1))
            action = agent.control(obs)
            obs, reward, done, info = env.step(action)
            
            if not isinstance(reward, float):
                reward = reward.item()
            total_reward += reward
        print(f'Episode {episode + 1}: Total Reward = {total_reward}')

def test():
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)
    ws_path = rospy.get_param('/ws_path')
    env = gym.make('URSimTracking-v0')
    
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed=seed)

    obs_dim, n_act = env.observation_space.shape[0], env.action_space.shape[0]
        
    tester = PPOGAEAgent(obs_dim, n_act, epochs=10, hdim=16, policy_lr=3e-3, value_lr=1e-3, max_std=1.0,
                            clip_range=0.2, seed=seed)
    tester.load_model(policy_weights_path='{}/models/model_z02_005/trained_policy_network_weights.h5'.format(ws_path),
                        value_weights_path='{}/models/model_z02_005/trained_value_network_weights.h5'.format(ws_path))
    run_test(env, tester)

if __name__ == '__main__':
    test()
