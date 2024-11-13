#!/usr/bin/env python3

from collections import deque
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# from sklearn.utils import shuffle
from ur_tracking.algorithm.PPOGAEAgent import PPOGAEAgent
# from ur_tracking.algorithm.ppo_gae import PPOGAEAgent
import rospy
import gym
from ur_tracking.env.ur_tracking_env import URSimTracking
from datetime import datetime

'''
PPO Agent with Gaussian policy
'''
def run_episode(env, agent, animate=False): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False

    n_step = 10000
    for update in range(n_step):
        obs = np.array(obs)
        obs = obs.astype(np.float32).reshape((1, -1)) # numpy.ndarray (1, num_obs)
        #print ("observes: ", obs.shape, type(obs)) # (1, 15)
        observes.append(obs)
        
        action = agent.get_action(obs) # List
        actions.append(action)
        obs, reward, done, info = env.step(action)
        
        if not isinstance(reward, float):
            reward = reward.item()
        rewards.append(reward) # List
        infos.append(info)

        if done is True:
            break
        
    return (np.concatenate(observes), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32), infos)

def run_policy(env, agent, episodes): # collect trajectories
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, infos = run_episode(env, agent) # numpy.ndarray
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos} 
        trajectories.append(trajectory)
    return trajectories
        
def add_value(trajectories, val_func): # Add value estimation for each trajectories
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.get_value(observes)
        trajectory['values'] = values

def add_gae(trajectories, gamma=0.95, lam=0.95): # Origin:gamma=0.99, lam=0.98 # generalized advantage estimation (for training stability)
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        # temporal differences
        tds = rewards + np.append(values[1:], 0) * gamma - values
        advantages = np.zeros_like(tds)
        advantage = 0
        for t in reversed(range(len(tds))):
            advantage = tds[t] + lam*gamma*advantage
            advantages[t] = advantage
        trajectory['advantages'] = advantages

def add_rets(trajectories, gamma=0.99): # compute the returns
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        
        returns = np.zeros_like(rewards)
        ret = 0
        for t in reversed(range(len(rewards))):
            ret = rewards[t] + gamma*ret
            returns[t] = ret            
        trajectory['returns'] = returns

def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])

    # Normalization of advantages 
    # In baselines, which is a github repo including implementation of PPO coded by OpenAI, 
    # all policy gradient methods use advantage normalization trick as belows.
    # The insight under this trick is that it tries to move policy parameter towards locally maximum point.
    # Sometimes, this trick doesnot work.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, returns

def plot_training_results(res_path, avg_return_list, avg_pol_loss_list, avg_val_loss_list, kl_divergence_list, entropy_list):
    fig, axs = plt.subplots(5, 1, figsize=(10, 25))
    
    axs[0].plot(avg_return_list)
    axs[0].set_title('Average Return over Training')
    axs[0].set_xlabel('Updates')
    axs[0].set_ylabel('Average Return')

    axs[1].plot(avg_pol_loss_list)
    axs[1].set_title('Policy Loss over Training')
    axs[1].set_xlabel('Updates')
    axs[1].set_ylabel('Policy Loss')
    
    axs[2].plot(avg_val_loss_list)
    axs[2].set_title('Value Loss over Training')
    axs[2].set_xlabel('Updates')
    axs[2].set_ylabel('Value Loss')
    
    axs[3].plot(kl_divergence_list)
    axs[3].set_title('KL Divergence over Training')
    axs[3].set_xlabel('Updates')
    axs[3].set_ylabel('KL Divergence')
    
    axs[4].plot(entropy_list)
    axs[4].set_title('Policy Entropy over Training')
    axs[4].set_xlabel('Updates')
    axs[4].set_ylabel('Entropy')
    
    plt.tight_layout()
    plt.savefig('{}.png'.format(res_path))
    plt.show()

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
    tester.load_model(policy_weights_path='{}/models/model/trained_policy_network_weights.h5'.format(ws_path),
                        value_weights_path='{}/models/model/trained_value_network_weights.h5'.format(ws_path))
    run_test(env, tester)
    
def main(): 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)
    ws_path = rospy.get_param('/ws_path')
    
    env = gym.make('URSimTracking-v0')
    seed = 0
    obs_dim = env.observation_space.shape[0] # 15
    n_act = env.action_space.shape[0] # 6
    agent = PPOGAEAgent(obs_dim, n_act, epochs=10, hdim=16, policy_lr=3e-3, value_lr=1e-3, max_std=1.0,
                        clip_range=0.2, seed=seed)
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed=seed)

    # avg_return_list = deque(maxlen=10)
    # avg_pol_loss_list = deque(maxlen=10)
    # avg_val_loss_list = deque(maxlen=10)
    avg_return_list = []
    avg_pol_loss_list = []
    avg_val_loss_list = []
    kl_divergence_list = []
    entropy_list = []


    episode_size = 10 # 10
    batch_size = 16
    nupdates = 20 # 500

    # save fig
    x_data = []
    y_data = []
    axes = plt.gca()
    axes.set_xlim(0, 350)
    axes.set_ylim(0, 1000)
    line, = axes.plot(x_data, y_data, 'r-')

    for update in range(nupdates+1):
        print("UPDATE: ", update)

        trajectories = run_policy(env, agent, episodes=episode_size)
        add_value(trajectories, agent)
        add_gae(trajectories)
        add_rets(trajectories)
        observes, actions, advantages, returns = build_train_set(trajectories)

        pol_loss, val_loss, kl, entropy = agent.update(observes, actions, advantages, returns, batch_size=batch_size)

        avg_pol_loss_list.append(pol_loss)
        avg_val_loss_list.append(val_loss)
        kl_divergence_list.append(kl)
        entropy_list.append(entropy)
        
        # avg_return_list.append([np.sum(t['rewards']) for t in trajectories])
        episode_returns = [np.sum(t['rewards']) for t in trajectories]
        avg_return = np.mean(episode_returns)
        avg_return_list.append(avg_return)
        
        x_data.append(update)
        y_data.append(np.mean(avg_return_list))
        
        if (np.mean(avg_return_list) > 1000): # Threshold return to success 
            print('[{}/{}] return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}'.format(update,nupdates, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break

    agent.save_model(policy_weights_path='{}models/trained_policy_network_weights.h5'.format(ws_path),
                     value_weights_path='{}models/trained_value_network_weights.h5'.format(ws_path))
    res_path = '{}figures/figure_{}'.format(ws_path, datetime.now().strftime('%Y-%m-%d_%H:%M'))
    plot_training_results(res_path, avg_return_list, avg_pol_loss_list, avg_val_loss_list, kl_divergence_list, entropy_list)

if __name__ == '__main__':
    main()
