from PIL import Image, ImageDraw
from dataloader import state_transform
from reinforcement import take_action, calculate_iou, find_positive_actions

def adjust_bbox_for_draw(bbox):
    bbox_new = (bbox[0], bbox[1], bbox[2]-1, bbox[3]-1)
    return bbox_new

def draw_boxes(state):
    image = state.image.copy()
    draw = ImageDraw.Draw(image)
    bbo = adjust_bbox_for_draw(state.bbox_observed)
    bbt = adjust_bbox_for_draw(state.bbox_true)
    draw.rectangle(bbo, outline=(0,255,255))
    draw.rectangle(bbt, outline=(255,0,255))
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

def localize2(state, max_n_actions, net):
    action_sequence = []
    vis = draw_boxes(state)
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
        vis_new.paste(draw_boxes(state), (vis.width, 0))
        vis = vis_new
        if done:
            break
    return vis, action_sequence

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