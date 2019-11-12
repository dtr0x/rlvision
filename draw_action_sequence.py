from dataloader import *
from reinforcement import *
from PIL import Image
import os

def draw_sequence(img_location, out_folder, obj_class):
    image = Image.open(img_location)
    action_history = torch.zeros(90)
    bbox = (0, 0, image.width, image.height)
    state = State(image, obj_class, bbox, action_history)
    done = False
    steps = 0
    while not done and steps < 40:
        state.image.crop(state.bbox).save(out_folder + "/{}.jpg".format(steps))
        a = find_best_action(state)
        if a is not None:
            reward, state, done = take_action(state, find_best_action(state))
        else:
            break
        steps += 1

img_dir = "high-res/train/car"
n = 0
for img_file in os.listdir(img_dir):
    out_folder = "demo/{}".format(n)
    os.mkdir(out_folder)
    img_location = os.path.join(img_dir, img_file)
    draw_sequence(img_location, out_folder, 0)
    n += 1

img_dir = "high-res/train/plane"
for img_file in os.listdir(img_dir):
    out_folder = "demo/{}".format(n)
    os.mkdir(out_folder)
    img_location = os.path.join(img_dir, img_file)
    draw_sequence(img_location, out_folder, 0)
    n += 1
