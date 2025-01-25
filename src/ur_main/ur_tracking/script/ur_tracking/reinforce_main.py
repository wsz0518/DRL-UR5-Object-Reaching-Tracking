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

def plot_training_results_old(res_path, avg_loss_list, avg_return_list):
    """
    将原先的 Loss 和 Return 绘制在同一个图的逻辑
    改为在两个独立的图形上分别绘制，生成并展示/保存两个图。
    """

    # 首先将 avg_return_list 内部的列表展开为标量形式
    avg_returns_scalar = [np.mean(r) for r in avg_return_list]

    # x 轴为更新次数
    x_data = range(len(avg_loss_list))

    fig1, ax1 = plt.subplots()
    ax1.plot(x_data, avg_loss_list)
    ax1.set_xlabel('Updates')
    ax1.set_ylabel('Average Policy Loss')
    ax1.tick_params(axis='y')
    ax1.set_title('Policy Loss over Training')
    fig1.tight_layout()
    fig1.savefig('{}_loss.png'.format(res_path))
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(x_data, avg_returns_scalar)
    ax2.set_xlabel('Updates')
    ax2.set_ylabel('Average Return')
    ax2.tick_params(axis='y')
    ax2.set_title('Average Return over Training')
    fig2.tight_layout()
    fig2.savefig('{}_return.png'.format(res_path))
    plt.show()

def plot_training_results(
    res_path,
    avg_loss_list,    # 对应策略 Loss
    avg_return_list   # 对应回报
):
    """
    将原先的Loss和Return绘制方式改为类似PPO的多图风格。
    这里只有两个指标 Policy Loss 和 Average Return。
    """

    # 将 avg_return_list 内部的列表转换成标量形式（若本身就是标量list可省略）
    avg_returns_scalar = [np.mean(r) for r in avg_return_list]

    # 1. Average Return
    plt.figure(figsize=(8, 8))
    plt.plot(avg_returns_scalar)
    plt.title('Average Return over Training from REINFORCE')
    plt.xlabel('Updates')
    plt.ylabel('Average Return')
    plt.tight_layout()
    plt.savefig('{}_avg_return.png'.format(res_path))
    plt.close()

    # 2. Policy Loss
    plt.figure(figsize=(8, 8))
    plt.plot(avg_loss_list)
    plt.title('Policy Loss over Training from REINFORCE')
    plt.xlabel('Updates')
    plt.ylabel('Policy Loss')
    plt.tight_layout()
    plt.savefig('{}_policy_loss.png'.format(res_path))
    plt.close()


def main():
    # Can check log msgs according to log_level {rospy.DEBUG, rospy.INFO, rospy.WARN, rospy.ERROR} 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)
    ws_path = rospy.get_param('/ws_path')
    
    env = gym.make('URSimTracking-v0')
    # env._max_episode_steps = 10000
    seed = 0
    obs_dim = env.observation_space.shape[0] # 15 # env.observation_space.shape[0]
    n_act = env.action_space.shape[0] # 6 #config: act_dim #env.action_space.n
    # Original agent = REINFORCEAgent(obs_dim, n_act, epochs=5, hdim=32, lr=3e-4,seed=seed)
    agent = REINFORCEAgent(obs_dim, n_act, epochs=10, hdim=16, lr=1e-5,seed=seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed) # tf1
    tf.random.set_seed(seed) # tf2
    env.seed(seed=seed)

    avg_return_list = [] #deque(maxlen=1000)
    avg_loss_list = [] #deque(maxlen=1000)

    episode_size = 5#5 # Original 1
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
