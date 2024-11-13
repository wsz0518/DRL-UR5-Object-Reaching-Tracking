#!/usr/bin/env python3

from collections import deque
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from ur_tracking.algorithm.REINFORCEAgent import REINFORCEAgent
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()

# import our training environment
import gym
from ur_tracking.env.ur_tracking_env import URSimTracking
import rospy
import rospkg


'''
PPO Agent with Gaussian policy
'''

def run_episode(env:URSimTracking, agent:REINFORCEAgent): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False

    for update in range(10000):
        print("episode step: ", update)
        action = agent.get_action([obs])

        next_obs, reward, done, info = env.step(action)
        
        observes.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)

        obs = next_obs

        if done is True:
            break

    return np.asarray(observes), np.asarray(actions), np.asarray(rewards), infos

def run_policy(env, agent, episodes): # collect trajectories. if 'evaluation' is true, then only mean value of policy distribution is used without sampling.
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        print("trajectory step: ", e)
        observes, actions, rewards, infos = run_episode(env, agent)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos}
        trajectories.append(trajectory)
    return trajectories

def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])

    return observes, actions, returns

def compute_returns(trajectories, gamma=0.995): # Add value estimation for each trajectories
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        returns = np.zeros_like(rewards)
        g = 0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma*g
            returns[t] = g
        trajectory['returns'] = returns


def main():
    # Can check log msgs according to log_level {rospy.DEBUG, rospy.INFO, rospy.WARN, rospy.ERROR} 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)
    
    env = gym.make('URSimReaching-v0')
    # env._max_episode_steps = 10000
    seed = 0
    obs_dim = env.observation_space.shape[0] # 15 # env.observation_space.shape[0]
    n_act = env.action_space.shape[0] # 6 #config: act_dim #env.action_space.n
    agent = REINFORCEAgent(obs_dim, n_act, epochs=5, hdim=32, lr=3e-4,seed=seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed=seed)

    avg_return_list = deque(maxlen=1000)
    avg_loss_list = deque(maxlen=1000)

    episode_size = 1
    batch_size = 16
    nupdates = 300 #100000

    for update in range(nupdates+1):
        print('update: ', update)
        trajectories = run_policy(env, agent, episodes=episode_size)
        compute_returns(trajectories)
        observes, actions, returns = build_train_set(trajectories)

        pol_loss = agent.update(observes, actions, returns, batch_size=batch_size)

        avg_loss_list.append(pol_loss)
        avg_return_list.append([np.sum(t['rewards']) for t in trajectories])
        
        if (update%1)==0:
            avg_loss = np.mean(avg_loss_list)
            avg_ret = np.mean(avg_return_list)
            print(avg_loss)
            print(avg_ret)
            print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update, nupdates, avg_loss, avg_ret))
            
        if (np.mean(avg_return_list) > 10000) and np.shape(np.mean(avg_loss_list)) == np.shape(np.mean(avg_return_list)): # Threshold return to success cartpole
            print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update,nupdates, np.mean(avg_loss_list), np.mean(avg_return_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break
	
    #env.close()

if __name__ == '__main__':
    main()