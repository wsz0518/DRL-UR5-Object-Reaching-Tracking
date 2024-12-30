#!/usr/bin/env python3

from collections import deque
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from ur_tracking.algorithm.REINFORCEAgent import REINFORCEAgent
# from ur_tracking.algorithm.reinforce import REINFORCEAgent
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()

# import our training environment
import gym
from ur_tracking.env.ur_tracking_env import URSimTracking
import rospy
import rospkg
from datetime import datetime


'''
PPO Agent with Gaussian policy
'''

def run_episode(env:URSimTracking, agent:REINFORCEAgent): # Run policy and collect (state, action, reward) pairs
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

    return observes, actions , returns

def compute_returns(trajectories, gamma=0.995): # Add value estimation for each trajectories
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        returns = np.zeros_like(rewards)
        g = 0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma*g
            returns[t] = g
        trajectory['returns'] = returns

def plot_training_results(res_path, avg_loss_list, avg_return_list):
    # 将 avg_return_list 内部的列表展开为单个列表
    # avg_return_list 中保存的是每次更新append([np.sum(t['rewards']) for t in trajectories])
    # 这里每次都是append一个列表，需先将其转化为均值等标量
    avg_returns_scalar = [np.mean(r) for r in avg_return_list]

    # 创建一个图形和两个子图
    fig, ax1 = plt.subplots()

    # x 轴为更新次数
    x_data = range(len(avg_loss_list))

    # 在左轴绘制平均LOSS
    color = 'tab:red'
    ax1.set_xlabel('Update Steps')
    ax1.set_ylabel('Average Policy Loss', color=color)
    ax1.plot(x_data, avg_loss_list, color=color, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 在同一张图中，但不同y轴上绘制平均回报
    ax2 = ax1.twinx()  # 共享x轴
    color = 'tab:blue'
    ax2.set_ylabel('Average Return', color=color)
    ax2.plot(x_data, avg_returns_scalar, color=color, label='Average Return')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # 调整子图间距
    plt.title('Training Progress')
    plt.savefig('{}.png'.format(res_path))
    plt.show()


def main():
    # Can check log msgs according to log_level {rospy.DEBUG, rospy.INFO, rospy.WARN, rospy.ERROR} 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)
    ws_path = rospy.get_param('/ws_path')
    
    env = gym.make('URSimTracking-v0')
    # env._max_episode_steps = 10000
    seed = 0
    obs_dim = env.observation_space.shape[0] # 15 # env.observation_space.shape[0]
    n_act = env.action_space.shape[0] # 6 #config: act_dim #env.action_space.n
    agent = REINFORCEAgent(obs_dim, n_act, epochs=5, hdim=32, lr=3e-4,seed=seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed) # tf1
    tf.random.set_seed(seed) # tf2
    env.seed(seed=seed)

    avg_return_list = deque(maxlen=1000)
    avg_loss_list = deque(maxlen=1000)

    episode_size = 1 # Original 1
    batch_size = 16
    nupdates = 50 #100000

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
            
        if (np.mean(avg_return_list) > 1000) and np.shape(np.mean(avg_loss_list)) == np.shape(np.mean(avg_return_list)): # Threshold return to success cartpole
            print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update,nupdates, np.mean(avg_loss_list), np.mean(avg_return_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break
	
    #env.close()
    agent.save_model(policy_weights_path='{}models_reinforce/trained_policy_weights.h5'.format(ws_path))
    res_path = '{}figures_reinforce/figure_{}'.format(ws_path, datetime.now().strftime('%Y-%m-%d_%H:%M'))
    plot_training_results(res_path, avg_loss_list, avg_return_list)

if __name__ == '__main__':
    main()
