import torch, torchvision
import cv2
from dataloader import *
import math
from numpy import argmax
from classifier.ResNet import ResNet

# Reinforcement learning actions and reward system

# The threshold for which an object detection is considered positive
CONFIDENCE_THRESHOLD = 0.8

# load the pre-trained classifier (trained on imagenet)
classifier = ResNet().to(device)
classifier.load_state_dict(torch.load("classifier/init_model.pth"))
classifier.eval()

# calculate the confidence score of the observed region (bounding box)
# from a given state. The confidence score returned is for the class of 
# object that the state image contains
def calculate_conf(state):
    img_observed = state.image.crop(state.bbox)
    img_t = transform(img_observed).unsqueeze(0).to(device)
    class_score = classifier(img_t)[state.obj_class]
    return class_score

# update an action history with a new encoded action
def update_action_history(action_history, action):
    action_history_new = action_history.clone()
    action_tmp = torch.zeros(9)
    action_tmp[action] = 1
    action = action_tmp
      
    last_actions = action_history_new[:81].clone()
       
    action_history_new[:9] = action
    action_history_new[9:] = last_actions
        
    return action_history_new

# take an action: the agent moves the bounding box of the state
# and receives a reward
def take_action(state, action):
    # unpack the state properties
    image, obj_class, bbox, action_history = state
    x1, y1, x2, y2 = bbox
    # a scalar factor alpha is used, 
    # based on the size of the current bounding box,
    # to determine how much the bounding is transformed 
    # after taking the action
    alph_w = int(0.2 * (x2 - x1))
    alph_h = int(0.2 * (y2 - y1))
    
    # the "trigger" action implies the search is over
    done = False
    
    if action == 0: #horizontal move to the right
        x1 += alph_w
        x2 = min(x2 + alph_w, image.width)
    elif action == 1: #horizontal move to the left
        x1 = max(x1 - alph_w, 0)
        x2 -= alph_w
    elif action == 2: #vertical move up
        y1 = max(y1 - alph_h, 0)
        y2 -= alph_h
    elif action == 3: #vertical move down
        y1 += alph_h
        y2 = min(y2 + alph_h, image.height)
    elif action == 4: #scale up
        x1 = max(x1 - math.floor(alph_w/2), 0)
        x2 = min(x2 + math.floor(alph_w/2), image.width)
        y1 = max(y1 - math.floor(alph_h/2), 0)
        y2 = min(y2 + math.floor(alph_h/2), image.height)
    elif action == 5: #scale down
        x1 += math.floor(alph_w/2)
        x2 -= math.floor(alph_w/2)
        y1 += math.floor(alph_h/2)
        y2 -= math.floor(alph_h/2)
    elif action == 6: #decrease height (aspect ratio)
        y1 += math.floor(alph_h/2)
        y2 -= math.floor(alph_h/2)
    elif action == 7: #decrease width (aspect ratio)
        x1 += math.floor(alph_w/2)
        x2 -= math.floor(alph_w/2)
    elif action == 8: #trigger
        done = True
        
    bbox_new = (x1, y1, x2, y2)
    action_history_new = update_action_history(action_history, action)
    next_state = State(image, obj_class, bbox_new, action_history_new)
    
    conf_old = calculate_conf(state)
    conf_new = calculate_conf(next_state)
    
    # if the trigger action is used, a large positive (negative) reward
    # is received if the confidence score of the area inside the bounding
    # box is above (below) the confidence threshold. 
    if done:
        if conf_new >= CONFIDENCE_THRESHOLD:
            reward = 3.0
        else:
            reward = -3.0
    # a reward of 1 is received if the next-state confidence score is higher 
    # than the previous state
    else:
        reward = torch.sign(conf_new - conf_old)
    
    return reward, next_state, done

# Find which actions produce a positive reward from current state 
def find_positive_actions(state):
    positive_actions = []
    for i in range(9):
        reward, next_state, done = take_action(state, i)
        if reward > 0:
            positive_actions.append(i)
    return positive_actions

# Find the best action from current state, based on the increase in confidence score
def find_best_action(state):
    confs = []
    if calculate_conf(state) >= CONFIDENCE_THRESHOLD:
        return 8
    for i in range(8):
        reward, next_state, done = take_action(state, i)
        confs.append(calculate_conf(next_state))
    best_next_state_conf_idx = argmax(confs)
    # If no action increases the confidence score, return None
    if calculate_conf(state) > confs[best_next_state_conf_idx]:
        return None
    return best_next_state_conf_idx
