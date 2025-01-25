import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import rospy
import pickle


ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
ABSOLUTETRAININGDATAPATH = rospy.get_param("/ur5/wrapper_training_results_path")
ABSOLUTETESTDATAPATH = rospy.get_param("/ur5/wrapper_test_results_path")

# read folder, find out json data
def plot_results(mode=None, win=None, training=False, texts=None, legend=True):
    if mode is None:
        return
    
    if training:
        folder_path = ABSOLUTETRAININGDATAPATH +'/monitor_' + mode
    else:
        folder_path = ABSOLUTETESTDATAPATH +'/monitor_' + mode
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
        if not training:
            plt.xlim(0, 9)
            plt.ylim(0, 300)
        plt.xlabel('Episode')
        plt.ylabel('Cumulated Episode Reward')
        plt.title('Rewards per Episode from ' + mode)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        if legend:
            plt.legend()
        plt.text(0.1, 0.2, texts, transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5))
        figure_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        if training:
            figure_path = (ABSOLUTETRAININGDATAPATH + '/figure_' + mode + '/{}.png').format(figure_name)
        else:
            figure_path = (ABSOLUTETESTDATAPATH + '/figure_' + mode + '/{}.png').format(figure_name)
        plt.savefig(figure_path)
        plt.show()
        
def save_Q_table(Q_values=None):
    if Q_values is None:
        return
    # df = pd.DataFrame(Q_values, index=[0])
    # data_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    # folder_path = ABSOLUTEDATAPATH+'/qtable_qlearn'
    # data_path = (folder_path + '/{}.csv').format(data_name)
    # df.to_csv(data_path,float_format='%.3f')
    
    data_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    folder_path = ABSOLUTETRAININGDATAPATH + '/qtable_qlearn'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_path = (folder_path + '/{}.json').format(data_name)

    Q_table_serializable = {str(k): v for k, v in Q_values.items()}
    with open(data_path, 'w') as f:
        json.dump(Q_table_serializable, f)

def save_Q_table_dict(Q_values=None):
    if Q_values is None:
        return
    data_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    folder_path = ABSOLUTETRAININGDATAPATH + '/qtable_qlearn'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_path = (folder_path + '/{}.pkl').format(data_name)
    with open(data_path, 'wb') as f:
        pickle.dump(Q_values, f)


def obs_to_state(obs:np.ndarray, format=None):
    if format is None:
        return obs
    elif format == '1f':
        return np.round(obs, 1)
    elif format == '2f':
        return np.round(obs, 2)
    elif format == '05':
        return np.round(obs * 20) / 20 #[1.00, 1.05 ,1.10, 1.15]