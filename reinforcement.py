import torch
import numpy as np
import cv2
from dataloader import State
import math
from PIL import Image, ImageDraw
from visualization import adjust_bbox_for_draw

def restart(state):
    # restart observed bounding box in one of the four corners
    w = state.image.width
    h = state.image.height
    w_box = int(0.75 * w)
    h_box = int(0.75 * h)
    if state.start_pos == 0:
        bbox_new = (0, 0, w_box, h_box) # top left
    elif state.start_pos == 1:
        bbox_new = (w-w_box, 0, w, h_box) # top right
    elif state.start_pos == 2: # bottom left
        bbox_new = (0, h-h_box, w_box, h)
    elif state.start_pos == 3: # bottom right
        bbox_new = (w-w_box, h-h_box, w, h)

    start_pos = (state.start_pos + 1) % 4
    # update action history with trigger action
    action_history_new = update_action_history(state.action_history, 8)
    state_new = State(draw_ior(state), bbox_new, state.bboxes_true, 
        action_history_new, state.n_trigger+1, start_pos)
    return state_new

def draw_ior(state):
    # draw inhibition of return cross over bounding box
    image = state.image.copy()
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = adjust_bbox_for_draw(state.bbox_observed)
    w = x2 - x1
    h = y2 - y1
    ior_width = int(0.25 * w)
    ior_height = int(0.25 * h)
    x_offset = int((w - ior_width)/2)
    y_offset = int((h - ior_height)/2)
    x1_ior = x1 + x_offset
    x2_ior = x2 - x_offset
    y1_ior = y1 + y_offset
    y2_ior = y2 - y_offset
    draw.rectangle((x1_ior, y1, x2_ior, y2), fill=0)
    draw.rectangle((x1, y1_ior, x2, y2_ior), fill=0)
    return image

def calculate_max_iou(state):
    # return maximum IOU from all bounding boxes
    image, bbox_observed, bboxes_true, action_history, n_trigger, start_pos = state
    img_mask = np.zeros((image.height, image.width))

    x1, y1, x2, y2 = bbox_observed
    img_mask[y1:y2, x1:x2] = 1.0

    max_iou = 0.0
    for gt_bbox in bboxes_true:
        gt_mask = np.zeros((image.height, image.width))
        x1, y1, x2, y2 = gt_bbox
        gt_mask[y1:y2, x1:x2] = 1.0
        img_and = cv2.bitwise_and(img_mask, gt_mask)
        img_or = cv2.bitwise_or(img_mask, gt_mask)
        j = np.count_nonzero(img_and)
        i = np.count_nonzero(img_or)
        iou_tmp = float(float(j)/float(i))
        if iou_tmp > max_iou:
            max_iou = iou_tmp
    
    return max_iou

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
    image, bbox_observed, bboxes_true, action_history, n_trigger, start_pos = state        
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

    if action == 8: # trigger 
        if calculate_max_iou(state) >= 0.6:
            reward = 3.0
        else:
            reward = -3.0
        next_state = restart(state) # updates state parameters
        if next_state.n_trigger == len(bboxes_true):
            done = True
    else:
        bbox_observed_new = (x1, y1, x2, y2)
        action_history_new = update_action_history(action_history, action)
        next_state = State(image, bbox_observed_new, bboxes_true, 
            action_history_new, n_trigger, start_pos)
    
        iou_old = calculate_max_iou(state)
        iou_new = calculate_max_iou(next_state)
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
