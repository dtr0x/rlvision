from PIL import Image, ImageDraw
from dataloader import state_transform
from reinforcement import *
import math

# Visualization functions to produce output from trained model

MAX_IMAGE_WIDTH = math.sqrt(2*Image.MAX_IMAGE_PIXELS)

# fix for PIL draw.rectangle
def adjust_bbox_for_draw(bbox):
    bbox_new = (bbox[0], bbox[1], bbox[2]-1, bbox[3]-1)
    return bbox_new

# return image from state with bounding box
def draw_boxes(state):
    image = state.image.copy()
    draw = ImageDraw.Draw(image)
    bbo = adjust_bbox_for_draw(state.bbox)
    draw.rectangle(bbo, outline=(0,255,255))
    return(image)

# draw actions taken from a given state to localize an object.
# A maximum number of actions and a trained network are provided as arguments.
def draw_localization_actions(state, max_n_actions, net):
    action_sequence = [] # action sequence
    conf_sequence = [] # sequence of confidence scores
    conf_sequence.append(calculate_conf(state))
    vis = draw_boxes(state)
    last_action_image = vis
    w = state.image.width
    h = state.image.height
    done = False
    for i in range(max_n_actions):
        img_t, action_history = state_transform([state])
        action = net(img_t, action_history).max(1).indices[0].item()
        action_sequence.append(action)
        reward, state, done = take_action(state, action)
        vis_new = Image.new('RGB', (vis.width + w, h))
        vis_new.paste(vis)
        last_action_image = draw_boxes(state)
        vis_new.paste(last_action_image, (vis.width, 0))
        vis = vis_new
        conf_sequence.append(calculate_conf(state))
        if done:
            break
    # return the last state image if the visualization sequence is too big
    if vis.width > MAX_IMAGE_WIDTH:
        vis = last_action_image
    return vis, action_sequence, conf_sequence