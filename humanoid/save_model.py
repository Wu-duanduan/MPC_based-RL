import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# self.state_all = torch.zeros(1, state_size)
# self.action_all = torch.zeros(1, action_size)
# self.next_state_all = torch.zeros(1, state_size)
# self.reward_all = torch.zeros(1, 1)

class Savemodel():

    def __init__(self, state_size, action_size):
        self.state_all = torch.zeros(1, 1)
        self.action_all = torch.zeros(1, 1)
        self.next_state_all = torch.zeros(1, 1)
        self.reward_all = torch.zeros(1, 1)

    def save(self, state, action, reward, next_state):
        self.state_all = torch.cat([self.state_all, torch.from_numpy(np.vstack([state])).float().to(device)], dim=1)
        self.action_all = torch.cat([self.action_all, torch.from_numpy(np.vstack([action])).float().to(device)], dim=1)
        self.next_state_all = torch.cat([self.next_state_all, torch.from_numpy(np.vstack([next_state])).float().
                                        to(device)], dim=1)
        self.reward_all = torch.cat([self.reward_all, torch.from_numpy(np.vstack([reward])).float().to(device)], dim=1)
        return self.state_all, self.action_all, self.reward_all, self.next_state_all
