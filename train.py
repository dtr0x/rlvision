# import packages and load dataset

import torchvision
import numpy as np
from PIL import Image
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
import random
import math
from collections import namedtuple
from itertools import count
import cv2

from reinforcement import *
from dqn import *
from dataloader import *
from optimization import *
from visualization import *

VOC2012 = torchvision.datasets.VOCDetection("drive/My Drive/VOC2012")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters / utilities

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(10000)

# training loop

train_loader = torch.utils.data.DataLoader(VOC2012, batch_size=BATCH_SIZE, collate_fn=default_collate, shuffle=True)

VOCtest = torchvision.datasets.VOCDetection("drive/My Drive/VOC2012", image_set='val')
test_loader = torch.utils.data.DataLoader(VOCtest, batch_size=1, collate_fn=default_collate, shuffle=True)
test_iter = enumerate(test_loader)

steps_done = 0

import timeit

def select_action(img_t, action_history, states):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # select best action from model with probability 1-epsilon
        with torch.no_grad():
            actions = policy_net(img_t, action_history)
            return torch.max(actions, 1).indices
    else:
        # select random positive action with probability epsilon
        actions = []
        for state in states:
            positive_actions = find_positive_actions(state)
            if len(positive_actions) > 0:
                action = random.choice(positive_actions)
            else:
                action = random.randrange(n_actions)
            actions.append(action)
        return torch.tensor(actions, device=device)

total_time = 0
print("First 10 DQN params (initialization):", policy_net.state_dict()['dqn.0.weight'][0][:10])
for i, states in enumerate(train_loader):
    print("Running batch", i)
    batch_steps = 0
    start = timeit.default_timer()
    while len(states) > 0 and batch_steps < 100:
        img_t, action_history = state_transform(states)
        actions = select_action(img_t, action_history, states)
        states_new = []
        for j in range(actions.shape[0]):
            action = actions[j].item()
            state = states[j]
            reward, next_state, done = take_action(state, action)
            reward = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward)
            if not done:
                states_new.append(next_state)
        optimize_model()
        
        if batch_steps % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        states = states_new
        batch_steps+=1
    
    # save visualization
    _, s = test_iter.__next__()
    localize(s[0], "img_{}".format(i))
    
    stop = timeit.default_timer()
    t = (stop-start)/60
    total_time += t
    print("Finished batch {0} in {1:.2f} minutes.".format(i, t))
    print("Total time: {0:.2f} minutes.".format(total_time))
    print("First 10 DQN params after batch {0}:".format(i), policy_net.state_dict()['dqn.0.weight'][0][:10])