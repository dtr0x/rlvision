from dataloader import *
from reinforcement import *
from PIL import Image
import os

cars_found = 0
for img_file in os.listdir("high-res/car"):
    image = Image.open("high-res/car/{}".format(img_file))
    obj_class = 0
    action_history = torch.zeros(90)
    bbox = (0, 0, image.width, image.height)
    state = State(image, obj_class, bbox, action_history)
    done = False
    steps = 0
    while not done and steps < 40:
        a = find_best_action(state)
        if a is not None:
            reward, state, done = take_action(state, find_best_action(state))
        else:
            break
        steps += 1
    if done and steps >= 5:
        print("Actions:", steps)
        cars_found += 1
        image.save("high-res/train/car/{}".format(img_file))
print("Car Images:", cars_found)

planes_found = 0
for img_file in os.listdir("high-res/plane"):
    image = Image.open("high-res/plane/{}".format(img_file))
    obj_class = 1
    action_history = torch.zeros(90)
    bbox = (0, 0, image.width, image.height)
    state = State(image, obj_class, bbox, action_history)
    done = False
    steps = 0
    while not done and steps < 40:
        a = find_best_action(state)
        if a is not None:
            reward, state, done = take_action(state, find_best_action(state))
        else:
            break
        steps += 1
    if done and steps >= 5:
        print("Actions:", steps)
        planes_found += 1
        image.save("high-res/train/plane/{}".format(img_file))
print("Plane Images:", planes_found)