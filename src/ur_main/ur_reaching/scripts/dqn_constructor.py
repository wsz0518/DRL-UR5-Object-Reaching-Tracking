import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
from utils import ABSOLUTETRAININGDATAPATH, ISOTIMEFORMAT


## create a 1-hidden-Network to approximate Q-Values
def create_Q_net(num_states, dim_mid, num_actions):
	
    class QNet(nn.Module):
        def __init__(self, num_states, dim_mid, num_actions):
            super().__init__()

            self.fc = nn.Sequential(
                nn.Linear(num_states, dim_mid),
                nn.ReLU(),
                nn.Linear(dim_mid, dim_mid),
                nn.ReLU(),
                nn.Linear(dim_mid, num_actions))

        def forward(self, x):
            x = self.fc(x)
            return x
    
    return QNet(num_states, dim_mid, num_actions)


## create a brain to choose actions using a updatable QNet
def create_brain(num_states, num_mid, num_actions, gamma, epsilon, lr):

    class Brain:
        def __init__(self, num_states, num_mid, num_actions, gamma, epsilon, lr):
            self.num_actions = num_actions
            self.epsilon = epsilon  # Original 1.0  
            self.gamma = gamma

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("self.device = ", self.device)
            
            torch.set_printoptions(precision=3)

            self.q_net = create_Q_net(num_states, num_mid, num_actions)
            self.q_net.to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        def updateQnet(self, obs_numpy, action, reward, next_obs_numpy):
            obs_tensor = torch.from_numpy(obs_numpy).float()
            obs_tensor.unsqueeze_(0)	
            obs_tensor = obs_tensor.to(self.device)

            next_obs_tensor = torch.from_numpy(next_obs_numpy).float()
            next_obs_tensor.unsqueeze_(0)
            next_obs_tensor = next_obs_tensor.to(self.device)

            self.optimizer.zero_grad()

            self.q_net.train() # train mode
            q = self.q_net(obs_tensor)

            with torch.no_grad():
                self.q_net.eval() # eval mode
                label = self.q_net(obs_tensor)
                next_q = self.q_net(next_obs_tensor)
                maxQ = np.max(next_q.cpu().detach().numpy(), axis=1)[0]
                
                '''with 6 outputs'''
                # if action >= 6:
                #     action -= 6
                
                label[:, action] = reward + self.gamma * maxQ
            # print(np.round(q.cpu().detach().numpy().flatten(), decimals=3))
            # print(np.round(label.cpu().detach().numpy().flatten(), decimals=3))
            loss = self.criterion(q, label)
            loss.backward()
            self.optimizer.step()

        def getAction(self, obs_numpy, is_training):
            obs_tensor = torch.from_numpy(obs_numpy).float()
            obs_tensor.unsqueeze_(0)
            obs_tensor = obs_tensor.to(self.device)
            with torch.no_grad():
                self.q_net.eval()
                q = self.q_net(obs_tensor)

                '''original'''
                action = np.argmax(q.cpu().detach().numpy(), axis=1)[0]
                # print(q)
                ## random choice
                if is_training and np.random.rand() < self.epsilon:
                    random_action = np.random.randint(self.num_actions)
                    print("rrrrrrrrrrr random choice:",random_action, " // ", action, " rrrrrrrrrrr")
                    action = random_action
                '''
                Was 6/12 outputs
                '''
            return action # np.random.randint(10, 12)

        def loadQnet(self, path):
            self.q_net.load_state_dict(torch.load(path, map_location=self.device))
            self.q_net.eval()

        def saveQnet(self, saving, path):
            if not saving:
                return
            data_name = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            data_path = path + '/{}.pth'.format(data_name)
            torch.save(self.q_net.state_dict(), data_path)
        
    return Brain(num_states, num_mid, num_actions, gamma, epsilon, lr)


## create a agent with brain of QNet to make decisions
def create_agent(num_states, num_mid, num_actions, gamma, epsilon, lr, q_net_path):

    class Agent:
        def __init__(self, num_states, num_mid, num_actions, gamma, epsilon, lr, q_net_path):
            self.brain = create_brain(num_states, num_mid, num_actions, gamma, epsilon, lr)
            self.q_net_path = q_net_path

        def updateQnet(self, obs, action, reward, next_obs):
            self.brain.updateQnet(obs, action, reward, next_obs)

        def getAction(self, obs, is_training):
            action = self.brain.getAction(obs, is_training)
            return action
        
        def loadQnet(self):
            self.brain.loadQnet(self.q_net_path)

        def saveQnet(self, saving, path=ABSOLUTETRAININGDATAPATH+'/model_dqn'):
            self.brain.saveQnet(saving, path)

    return Agent(num_states, num_mid, num_actions, gamma, epsilon, lr, q_net_path)

'''
def getAction(self, obs_numpy, is_training):
            ## random action choice
            if is_training and np.random.rand() < self.epsilon:
                # Original random chioce
                print("rrrrrrrrrrr random Action choice rrrrrrrrrrr")
                action = np.random.randint(self.num_actions)
            else:
                obs_tensor = torch.from_numpy(obs_numpy).float()
                obs_tensor.unsqueeze_(0)
                obs_tensor = obs_tensor.to(self.device)
                with torch.no_grad():
                    self.q_net.eval()
                    q = self.q_net(obs_tensor)
                    # qs = q.cpu().detach().numpy().flatten()
                    # minQ, maxQ = np.min(qs), np.max(qs)
                    # mag = np.maximum(np.abs(minQ), np.abs(maxQ))
                    # qs = [qs[i] + np.random.rand() * mag - .5 * mag for i in range(self.num_actions)]
                    # qs = np.array(qs)
                    # action = np.argmax(qs)
                    action = np.argmax(q.cpu().detach().numpy(), axis=1)[0]

            if is_training and self.epsilon > self.eb:
                self.epsilon *= self.ed
            return action
'''

'''with 6 outputs'''
# action = np.argmax(q.cpu().detach().numpy(), axis=1)[0]
# if info:
#     action += 6
# if is_training and np.random.rand() < self.epsilon:
#     print("rrrrrrrrrrr random choice rrrrrrrrrrr")
#     action = np.random.randint(self.num_actions)
#     if info:
#         action += 6
        
'''with 12 outputs'''
# if info is False:
#     action = np.argmax(q.cpu().detach().numpy()[:,:6], axis=1)[0]
#     ## random choice
#     if is_training and np.random.rand() < self.epsilon:
#         random_action = np.random.randint(6)# self.num_actions
#         print("rrrrrrrrrrr random choice:",random_action, " // ", action, " rrrrrrrrrrr")
#         action = random_action
# else:
#     action = np.argmax(q.cpu().detach().numpy()[:,6:], axis=1)[0] + 6
#     ## random choice
#     if is_training and np.random.rand() < self.epsilon:
#         random_action = np.random.randint(6,12) # self.num_actions
#         print("rrrrrrrrrrr random choice:",random_action, " // ", action, " rrrrrrrrrrr")
#         action = random_action