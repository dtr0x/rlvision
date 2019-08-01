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
    vis.save("drive/My Drive/visualization/{}.png".format(name))