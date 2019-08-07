from PIL import Image, ImageDraw
from dataloader import state_transform
from reinforcement import take_action

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
    vis = draw_boxes(state)
    w = state.image.width
    h = state.image.height
    for i in range(20):
        img_t, action_history = state_transform([state])
        action = net(img_t, action_history).max(1).indices[0].item()
        reward, state, done = take_action(state, action)
        vis_new = Image.new('RGB', (vis.width + w, h))
        vis_new.paste(vis)
        vis_new.paste(draw_boxes(state), (vis.width, 0))
        vis = vis_new
        if done:
            break
    vis.save("visualization/{}.png".format(img_name))