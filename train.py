# import packages and load dataset

import numpy as np
from PIL import Image
from IPython.display import display
import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import random
import math

from itertools import count
import cv2

from reinforcement import *
from dqn import *
from dataloader import *
from visualization import *

VOC2012 = torchvision.datasets.VOCDetection("VOC2012", image_set='train')

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

import torch.nn.functional as F

def get_last_action(state):
    last_action = state.action_history[:9]
    return last_action.nonzero().item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: get_last_action(s) != 8,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = [s for s in batch.next_state if get_last_action(s) != 8]
    non_final_img_t, non_final_action_history = state_transform(non_final_next_states)
    
    state_batch = batch.state
    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.cat(batch.reward)

    img_t, action_history = state_transform(state_batch)
    
    actions = policy_net(img_t, action_history)

    state_action_values = policy_net(img_t, action_history).gather(1, action_batch.view(-1, 1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_img_t, non_final_action_history).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp(-1, 1)
    optimizer.step()

# Visualization

from PIL import ImageDraw

def draw_boxes(state):
    image = state.image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(state.bbox_true, outline=(255,0,255))
    draw.rectangle(state.bbox_observed, outline=(0,255,255))
    return(image)

def localize(state, name):
    vis = draw_boxes(state)
    w = state.image.width
    h = state.image.height
    for i in range(20):
        img_t, action_history = state_transform([state])
        action = policy_net(img_t, action_history).max(1).indices[0].item()
        reward, state, done = take_action(state, action)
        vis_new = Image.new('RGB', (vis.width + w, h))
        vis_new.paste(vis)
        vis_new.paste(draw_boxes(state), (vis.width, 0))
        vis = vis_new
        if done:
            break
    vis.save("visualization/{}.png".format(name))

# training loop

train_loader = torch.utils.data.DataLoader(VOC2012, batch_size=BATCH_SIZE, collate_fn=default_collate, shuffle=True)

VOCtest = torchvision.datasets.VOCDetection("VOC2012", image_set='val')
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
