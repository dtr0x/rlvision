import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from reinforcement import *
from SingleClassDetection import *
from visualization import draw_localization_actions
import os, sys, time, random

best_model = torch.load("models_iou/target_net_276.pt").to(device)

# test data
class_name = "aeroplane"
VOCtrain = SingleClassDetection(class_name, 'train')
print("Test set size: {}".format(len(VOCtrain)))
train_loader = torch.utils.data.DataLoader(VOCtrain, 
    batch_size=1, collate_fn=default_collate)

max_actions = 40

for i, [s] in enumerate(train_loader):
    action_sequence = []
    conf_sequence = []
    conf_sequence.append(calculate_conf(s).item())
    for j in range(max_actions):
        a = find_best_action(s)
        if a is None:
            a = random.randrange(8)
        action_sequence.append(a)
        reward, s, done = take_action(s, a)
        if done:
            if reward > 0:
                print("Found object {} in {} steps with confidence {}.".format(i, j+1, conf_sequence[-1]))
            break
        conf_sequence.append(calculate_conf(s).item())
    if not done or reward < 0:
    	print("Couldn't find object {}.".format(i))

    print("Confidence sequence:", conf_sequence)
    print("Action sequence:", action_sequence)
    #vis, action_sequence, conf_sequence = draw_localization_actions(s, max_actions, best_model)
    #actions_taken = len(action_sequence)
    #if actions_taken == max_actions:
    #    print("Could not localize item {}.".format(i))
    #    print([conf.item() for conf in conf_sequence])
    #if conf_sequence[-1] < 0.8:
    #    print("Localization for item {} failed with confidence < 0.8.".format(i))
    #    print([conf.item() for conf in conf_sequence])
    #else:
    #    print("Localized item {} in {} actions.".format(i, actions_taken))
    #    print([conf.item() for conf in conf_sequence])
