import torch
import numpy as np
import cv2
from dataloader import *
import math

imagenet_classes = eval(open("imagenet_classes.txt").read())

def calculate_conf(state, classifier):
    img_observed = state.image.crop(state.bbox_observed)
    img_t = transform(img_observed).unsqueeze(0).to(device)
    class_scores = torch.nn.functional.softmax(classifier(img_t), dim=1)
    return imagenet_classes[class_scores.argmax().item()], class_scores.max().item()

def calculate_iou(state):
    image, bbox_observed, bbox_true, action_history = state

    img_mask = np.zeros((image.height, image.width))
    gt_mask = np.zeros((image.height, image.width))

    x1, y1, x2, y2 = bbox_observed
    img_mask[y1:y2, x1:x2] = 1.0

    x1, y1, x2, y2 = bbox_true
    gt_mask[y1:y2, x1:x2] = 1.0

    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    
    return iou

def update_action_history(action_history, action):
    action_history_new = action_history.clone()
    action_tmp = torch.zeros(9)
    action_tmp[action] = 1
    action = action_tmp
      
    last_actions = action_history_new[:81].clone()
       
    action_history_new[:9] = action
    action_history_new[9:] = last_actions
        
    return action_history_new
 
def take_action(state, action):
    image, bbox_observed, bbox_true, action_history = state        
    x1, y1, x2, y2 = bbox_observed
    alph_w = int(0.2 * (x2 - x1))
    alph_h = int(0.2 * (y2 - y1))
    
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
        
    bbox_observed_new = (x1, y1, x2, y2)
    action_history_new = update_action_history(action_history, action)
    next_state = State(image, bbox_observed_new, bbox_true, action_history_new)
    
    iou_old = calculate_iou(state)
    iou_new = calculate_iou(next_state)
       
    if done:
        if iou_new >= 0.6:
            reward = 3.0
        else:
            reward = -3.0
    else:
        reward = np.sign(iou_new - iou_old)
        
    return reward, next_state, done

def find_positive_actions(state):
    positive_actions = []
    for i in range(9):
        reward, next_state, done = take_action(state, i)
        if reward > 0:
            positive_actions.append(i)
    return positive_actions

def find_best_actions(state):
    iou_diff = []
    if calculate_iou(state) >= 0.6:
        return [8]
    for i in range(8):
        reward, next_state, done = take_action(state, i)
        iou_old = calculate_iou(state)
        iou_new = calculate_iou(next_state)
        iou_diff.append(iou_new - iou_old)
    return np.argwhere(iou_diff == np.max(iou_diff)).flatten().tolist()
