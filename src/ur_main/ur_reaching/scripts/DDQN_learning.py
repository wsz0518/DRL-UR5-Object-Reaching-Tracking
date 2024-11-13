#!/usr/bin/env python3

import rospy
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty

cart_pose = Pose()
pole_pose = Pose()
pole_twist = Twist()
y_angular = 0
cart_pose_x = 0

cart_pose = 0
cart_pose_x = 0
y_angular = 0
cart_vel_x = 0

pub_cart = rospy.Publisher('/cart_controller/command', Float64, queue_size = 10)
reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

obs_num = 4
acts_num = 10
total_rewards = []
number_of_steps = []

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99  
MAX_STEPS = 200
NUM_EPISODES = 500

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)
        print(self.main_q_network)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        # 1. check the batch size
        if len(self.memory) < BATCH_SIZE:
            return
        # 2. create a mini batch
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        # 3. find target value Q(s_t, a_t)
        self.expected_state_action_values = self.get_expected_state_action_values()
        # 4. update weights
        self.update_main_q_network()

    def decide_action(self, state, episode):

        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)

        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]) 

        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):

        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):

        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

def get_cart_pose(data):
    global cart_pose_x, y_angular, cart_vel_x
    ind = data.name.index('cart_pole::cart_link')
    cart_pose = data.pose[ind]
    cart_vel = data.twist[ind]

    ind_pitch = data.name.index('cart_pole::pole_link')
    pole_twist = data.twist[ind_pitch]

    cart_pose_x = cart_pose.position.x
    cart_vel_x = cart_vel.linear.x
    y_angular = pole_twist.angular.y

def simulate(episode):
    global y_angular, cart_pose_x, cart_vel_x

    yaw_angle = 0
    done = False
    time_interval = 0.02
    
    rospy.wait_for_service('/gazebo/reset_simulation')
    reset_simulation_client()

    observation_next = np.array([0, 0, 0, 0], dtype='float16')
    state = np.array([0, 0, 0, 0], dtype='float16')
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = torch.unsqueeze(state, 0)
    episode_10_list = np.zeros(10)
    step = 0

    for step in range(MAX_STEPS):
        time1 = time.time()
        yaw_angle += y_angular*time_interval
        observation_next[0] = cart_pose_x
        observation_next[1] = cart_vel_x
        observation_next[2] = yaw_angle 
        observation_next[3] = y_angular

        if(step == 0):
            observation_next[0] = 0
            observation_next[1] = 0
            observation_next[2] = 0
            observation_next[3] = 0
        
        action = agent.get_action(state, episode) 
        force = action*16/9 - 8
        pub_cart.publish(force)

        if(abs(yaw_angle) > 0.6 or abs(cart_pose_x) > 1.0 or step == MAX_STEPS):
            done = True

        if done:
            state_next = None

            episode_10_list = np.hstack(
                (episode_10_list[1:], step + 1))

            if step < 195:
                reward = torch.FloatTensor([-1.0])
            else:
                reward = torch.FloatTensor([1.0])
        else:
            reward = torch.FloatTensor([0.0])
            state_next = observation_next
            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
            state_next = torch.unsqueeze(state_next, 0)

        agent.memorize(state, action, state_next, reward)
        agent.update_q_function()

        state = state_next

        if done:
            print('Episode:%d Finished after %d steps, average for 10 episodes = %.1lf' % (
                episode, step + 1, episode_10_list.mean()))

            if(episode % 2 == 0):
                agent.update_target_q_function()

            plot_durations(step)
            break        

        time2 = time.time()
        interval = time2 - time1
        if(interval < time_interval):
            time.sleep(time_interval - interval)

def plot_durations(step):
    plt.figure(2)
    plt.clf()
    number_of_steps.append(step)
    x = np.arange(0, len(number_of_steps))
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(x, number_of_steps)

    plt.pause(0.001)

if __name__ == '__main__':
    rospy.init_node('cart_pole_simulator', anonymous=True)
    rospy.Subscriber("/gazebo/link_states", LinkStates, get_cart_pose)

    num_states = 4
    num_actions = 10
    agent = Agent(num_states, num_actions)

    for episode in range(NUM_EPISODES): 
        simulate(episode)

    x = np.arange(0, len(number_of_steps))
    plt.plot(x, number_of_steps)
    plt.show()
