import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
from visualization import *
from dataloader import *
import os

pr = np.load('precision_and_recall.npz')
p = pr['precision']
r = pr['recall']

f1score = 2*(p*r)/(p+r)
f1score[np.isnan(f1score)] = 0

plt.plot(f1score)
plt.xlabel("epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score Per Epoch")
plt.savefig(os.path.join("plots/all", "f1score.png"), bbox_inches="tight")
plt.clf()

best_model_idx = np.argmax(f1score)
best_model = \
torch.load("models/target_net_{}.pt".format(best_model_idx)).to(device)
print("Best Model: ", best_model_idx)

VOCtest = torchvision.datasets.VOCDetection("VOC2012", image_set='val')
test_loader = torch.utils.data.DataLoader(VOCtest, batch_size=1, collate_fn=default_collate)

n_items = len(VOCtest)
n_success = 0
n_max_actions = 0
n_false_trigger = 0
n_actions = []

for i, [s] in enumerate(test_loader):
	vis, action_sequence, iou = localize2(s, 100, best_model)
	n_action = len(action_sequence)
	if n_action == 100:
		print("Localization failed for item {}, reached max steps.".format(i))
		n_max_actions += 1
	elif iou < 0.5:
		print("Localization failed for item {} with < 50% IOU.".format(i))
		n_false_trigger +=1
	else:
		vis.save("visualization/all/{}.png".format(i))
		print("Localized item {0} in {1} actions.".format(i, n_action))
		n_actions.append(n_action)
		n_success += 1

mean_actions = np.mean(n_actions)

print("Successfully localized {0} of {1} items \
with {2:.2f} average number of actions taken.".format(n_success, n_items, mean_actions))
print("Reached action limit for {} items.".format(n_max_actions))
print("False Positives (triggers with < 50% IOU): {}".format(n_false_trigger))