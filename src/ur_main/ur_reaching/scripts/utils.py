import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import rospy


ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
ABSOLUTEDATAPATH = rospy.get_param("/ur5/wrapper_results_path")

# read folder, find out json data
def plot_results(mode=None, win=None):
    if mode is None:
        return
    
    folder_path = ABSOLUTEDATAPATH +'/monitor_' + mode
    # folder_path also can be monitor_dqn
    files = os.listdir(folder_path)
    file_name = None
    for f in files:
        if f[10:17] == 'episode':
            file_name = f
    if not files or file_name is None:
        return
    
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as f:
        data = json.load(f)

        rewards = data['episode_rewards']
        episodes = [episode for episode in range(len(rewards))]

        plt.plot(episodes, rewards, label='Episode Rewards', color='blue', linewidth=1)
        # add moving average
        if win is not None:
            rewards_series = pd.Series(rewards)
            moving_average = rewards_series.rolling(window=win).mean()
            plt.plot(episodes, moving_average,label='Moving Average', color='red', linewidth=1)

        # # add episode type
        # for x, y, t in zip(episodes, rewards, types):
        #     plt.text(x, y+0.3, t, ha='center', va='bottom')

        plt.xlabel('Episode')
        plt.ylabel('Cumulated Episode Reward')
        plt.title('Rewards per Episode')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        figure_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        figure_path = (ABSOLUTEDATAPATH + '/figure_' + mode + '/{}.png').format(figure_name)
        plt.savefig(figure_path)
        plt.show()
        
def save_Q_table(folder_path=ABSOLUTEDATAPATH+'/qtable_qlearn', Q_values=None, idx=0):
    if Q_values is None:
        return
    df = pd.DataFrame(Q_values, index=[idx])
    data_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    data_path = (folder_path + '/{}.csv').format(data_name)
    df.to_csv(data_path,float_format='%.3f')

def obs_to_state(obs:np.ndarray, format=None):
    if format is None:
        return obs
    elif format == '1f':
        return np.round(obs, 1)
    elif format == '2f':
        return np.round(obs, 2)
    elif format == '05':
        return np.round(obs * 20) / 20 #[1.00, 1.05 ,1.10, 1.15]