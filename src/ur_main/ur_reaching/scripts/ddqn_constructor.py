import numpy as np
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
CAPACITY = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.0001
MAX_STEPS = 200
NUM_EPISODES = 500


## create a replay memory to carry experience
def create_replay_memory(capacity):
    
    class ReplayMemory:

        def __init__(self, capacity):
            self.capacity = capacity
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
    
    return ReplayMemory(capacity)


## create Neural Network for Online & Target Network
def create_net(n_in, n_mid, n_out):

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
        
    return Net(n_in, n_mid, n_out)


## create the brain of an agent
def create_brain(num_states, num_mid, num_actions, gamma, batch_size, capacity, lr=0.0001):
    
    class Brain:
    
        def __init__(self, num_states, num_mid, num_actions, gamma, batch_size, capacity, lr):
            self.num_actions = num_actions 
            self.gamma = gamma
            self.batch_size = batch_size
            self.memory = create_replay_memory(capacity)
            self.main_q_network = create_net(num_states, num_mid, num_actions)
            self.target_q_network = create_net(num_states, num_mid, num_actions)
            print(self.main_q_network)
            self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=lr)

        def replay(self):
            # 1. check the batch size
            if len(self.memory) < self.batch_size:
                return
            # 2. create a mini batch
            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
            # 3. find target value Q(s_t, a_t)
            self.expected_state_action_values = self.get_expected_state_action_values()
            # 4. update weights
            self.update_main_q_network()

        def decide_action(self, state, episode):
            ## calculate epsilon for episodes
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
            transitions = self.memory.sample(self.batch_size)

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

            # non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
            #                                             self.batch.next_state)))
            non_final_mask = torch.tensor([s is not None for s in self.batch.next_state],
                                          dtype=torch.bool)
            next_state_values = torch.zeros(self.batch_size)

            a_m = torch.zeros(self.batch_size).type(torch.LongTensor)

            a_m[non_final_mask] = self.main_q_network(
                self.non_final_next_states).detach().max(1)[1]

            a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

            next_state_values[non_final_mask] = self.target_q_network(
                self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

            expected_state_action_values = self.reward_batch + self.gamma * next_state_values

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
    
    return Brain(num_states, num_mid, num_actions, gamma, batch_size, capacity, lr)


## create an agent to training DDQN
def create_agent(num_states, num_mid, num_actions, gamma, batch_size, capacity, lr):

    class Agent:
        def __init__(self, num_states, num_mid, num_actions, gamma, batch_size, capacity, lr):
            self.brain = create_brain(num_states, num_mid, num_actions, gamma, batch_size, capacity, lr)

        def update_q_function(self):
            self.brain.replay()

        def getAction(self, state, episode):
            state = self.tensorize(state)
            action = self.brain.decide_action(state, episode)
            return action

        def memorize(self, state, action, state_next, reward):
            state = self.tensorize(state)
            state_next = self.tensorize(state_next)
            reward = self.tensorize(reward)
            self.brain.memory.push(state, action, state_next, reward)

        def update_target_q_function(self):
            self.brain.update_target_q_network()

        def tensorize(self, data):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).type(torch.FloatTensor)
            elif np.issubdtype(type(data), np.floating):
                data = torch.tensor(data)
            else: raise TypeError

            return torch.unsqueeze(data, 0)
    
    return Agent(num_states, num_mid, num_actions, gamma, batch_size, capacity, lr)
