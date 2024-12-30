#!/usr/bin/env python3

import random
import ast

class QLearn:

    """
    Simple Q-Learning implementation.
    @author: Victor Mayoral Vilches <victor@erlerobotics.com>
    """

    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        # self.coarse_positioning_actions = self.actions[0:6]
        # self.fine_positioning_actions = self.actions[6:]

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0) # default Qvalue

    def learnQ(self, state, action, reward, value):
        """
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        q = [self.getQ(state, a) for a in self.actions]
        # print("Q for this state:", np.round(np.asarray(q), decimals=1))
        maxQ = max(q)
        if random.random() < self.epsilon:
            print("rrrrrrrrrrr random Action choice rrrrrrrrrrr")
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)
            # return random.randint(0, 5)
        count = q.count(maxQ)
        # In case there're several state-action max values, we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ] # [0,1,2,3,4,5] if maxQ=0.0
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        action = self.actions[i]
        return action
    
    def chooseTestAction(self, state):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        count = q.count(maxQ)
        # In case there're several state-action max values, we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ] # [0,1,2,3,4,5] if maxQ=0.0
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)



# def chooseAction(self, state):
    # if info is False: # coarse
        #     q = [self.getQ(state, a) for a in self.coarse_positioning_actions]
        #     # print("Q for this state:", np.round(np.asarray(q), decimals=1))
        #     maxQ = max(q)
        #     if random.random() < self.epsilon:
        #         print("rrrrrrrrrrr random Action choice rrrrrrrrrrr")
        #         minQ = min(q); mag = max(abs(minQ), abs(maxQ))
        #         # add random values to all the actions, recalculate maxQ
        #         q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.coarse_positioning_actions))]
        #         maxQ = max(q)
        #         # return random.randint(0, 5)
        #     count = q.count(maxQ)
        #     # In case there're several state-action max values, we select a random one among them
        #     if count > 1:
        #         best = [i for i in range(len(self.coarse_positioning_actions)) if q[i] == maxQ] # [0,1,2,3,4,5] if maxQ=0.0
        #         i = random.choice(best)
        #     else:
        #         i = q.index(maxQ)
        #     action = self.coarse_positioning_actions[i]
        # else: # info=True, fine
        #     q = [self.getQ(state, a) for a in self.fine_positioning_actions]
        #     maxQ = max(q)
        #     if random.random() < self.epsilon:
        #         print("rrrrrrrrrrr random Action choice rrrrrrrrrrr")
        #         minQ = min(q); mag = max(abs(minQ), abs(maxQ))
        #         # add random values to all the actions, recalculate maxQ
        #         q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.fine_positioning_actions))]
        #         maxQ = max(q)
        #         # return random.randint(0, 5)
        #     count = q.count(maxQ)
        #     # In case there're several state-action max values, we select a random one among them
        #     if count > 1:
        #         best = [i for i in range(len(self.fine_positioning_actions)) if q[i] == maxQ] # [0,1,2,3,4,5] if maxQ=0.0
        #         i = random.choice(best)
        #     else:
        #         i = q.index(maxQ)
        #     action = self.fine_positioning_actions[i]