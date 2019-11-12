from PIL import Image, ImageDraw, ImageFilter
from dataloader import state_transform
from reinforcement import *
import math

MAX_IMAGE_WIDTH = math.sqrt(2*Image.MAX_IMAGE_PIXELS)

def adjust_bbox_for_draw(bbox):
    bbox_new = (bbox[0], bbox[1], bbox[2]-1, bbox[3]-1)
    return bbox_new

def draw_boxes(state):
    image = state.image.copy()
    bbo = adjust_bbox_for_draw(state.bbox_observed)
    bbt = adjust_bbox_for_draw(state.bbox_true)
    ob_region = image.crop(bbo)
    gt_region = image.crop(bbt)
    blur_img = image.filter(ImageFilter.GaussianBlur(radius=10))
    blur_gt = gt_region.filter(ImageFilter.GaussianBlur(radius=4))
    blur_img.paste(blur_gt, bbt)
    draw = ImageDraw.Draw(blur_img)
    blur_img.paste(ob_region, bbo)
    draw.rectangle(bbo, outline=(255,0,0))
    draw.rectangle(bbt, outline=(0,0,0))
    return(blur_img)

def draw_boxes_conf(state, classifier):
    image = state.image.copy()
    draw = ImageDraw.Draw(image)
    bbo = adjust_bbox_for_draw(state.bbox_observed)
    bbt = adjust_bbox_for_draw(state.bbox_true)
    draw.rectangle(bbo, outline=(0,255,255))
    draw.rectangle(bbt, outline=(255,0,255))
    class_score = calculate_conf(state, classifier)
    draw.text((0, 0), 
        class_score[0] + ": {:.2f}".format(class_score[1]), (255,0,0))
    return(image)

def localize(state, img_name, net):
    action_sequence = []
    vis = draw_boxes(state)
    w = state.image.width
    h = state.image.height
    done = False
    for i in range(40):
        img_t, action_history = state_transform([state])
        action = net(img_t, action_history).max(1).indices[0].item()
        action_sequence.append(action)
        reward, state, done = take_action(state, action)
        vis_new = Image.new('RGB', (vis.width + w, h))
        vis_new.paste(vis)
        vis_new.paste(draw_boxes(state), (vis.width, 0))
        vis = vis_new
        if done:
            break
    vis.save("visualization/{}.png".format(img_name))
    return action_sequence

def draw_localization_actions(state, max_n_actions, net):
    action_sequence = []
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
        if done:
            break
    iou = calculate_iou(state)
    if vis.width > MAX_IMAGE_WIDTH:
        vis = last_action_image
    return vis, action_sequence, iou

def draw_localization_actions_gif(state, max_n_actions, net):
    action_sequence = []
    w = state.image.width
    h = state.image.height
    frames = []
    frames.append(draw_boxes(state))
    done = False
    for i in range(max_n_actions):
        img_t, action_history = state_transform([state])
        action = net(img_t, action_history).max(1).indices[0].item()
        action_sequence.append(action)
        reward, state, done = take_action(state, action)
        frame = draw_boxes(state)
        if done:
            draw = ImageDraw.Draw(frame)
            draw.text((0,0), "{} actions taken.".format(len(frames)), (0,0,0))
            frames.append(frame)
            break
        frames.append(frame)
    iou = calculate_iou(state)
    return frames, action_sequence, iou

def draw_sequence_with_conf_score(state, max_n_actions, net, classifier):
    action_sequence = []
    vis = draw_boxes_conf(state, classifier)
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
        last_action_image = draw_boxes_conf(state, classifier)
        vis_new.paste(last_action_image, (vis.width, 0))
        vis = vis_new
        if done:
            break
    iou = calculate_iou(state)
    if vis.width > MAX_IMAGE_WIDTH:
        vis = last_action_image
    return vis, action_sequence, iou

def draw_action_sequence(state, action_sequence, img_name):
    vis = draw_boxes(state)
    print("IOU:", calculate_iou(state))
    print("Positive Actions:", find_positive_actions(state))
    w = state.image.width
    h = state.image.height
    for action in action_sequence:
        reward, state, done = take_action(state, action)
        print("IOU:", calculate_iou(state))
        print("Positive Actions:", find_positive_actions(state))
        vis_new = Image.new('RGB', (vis.width + w, h))
        vis_new.paste(vis)
        vis_new.paste(draw_boxes(state), (vis.width, 0))
        vis = vis_new
    vis.save("visualization/action_sequences/{}.png".format(img_name))