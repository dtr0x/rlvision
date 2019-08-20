import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from reinforcement import take_action, calculate_iou
from PlaneDetection import PlaneDetection
import os

VOCtest = PlaneDetection('val')
test_loader = torch.utils.data.DataLoader(VOCtest, batch_size=1, collate_fn=default_collate)

def localize(state, net):
    n_actions = 0
    iou = None
    done = False
    for i in range(100):
        img_t, action_history = state_transform([state])
        action = net(img_t, action_history).max(1).indices[0].item()
        reward, state, done = take_action(state, action)
        n_actions += 1 
        if done:
            iou = calculate_iou(state)
            break
    return n_actions, iou

def precision_and_recall(net):
    tp = 0
    fp = 0
    fn = 0
    actions_hist = []
    for i, [s] in enumerate(test_loader):
        n_actions, iou = localize(s, net)
        actions_hist.append(n_actions)
        if iou is not None:
            if iou >= 0.5:
                 tp += 1
            else:
                fp +=1
        else:
            fn +=1
    if tp > 0 or fp > 0:
        precision = tp/(tp+fp)
    else:
        precision = 0
    if tp > 0 or fn > 0:
        recall = tp/(tp+fn)
    else:
        recall = 0
    return {'p': precision, 'r': recall, 'actions_hist': actions_hist}

def main():
    precision = []
    recall = []
    model_path = "models"
    n_models = len(os.listdir(model_path))
    for i in range(48, 52):
        print("Evaluating model {}...".format(i))
        model = os.path.join(model_path, "target_net_{}.pt".format(i))
        net = torch.load(model).to(device)
        pr = precision_and_recall(net)
        precision.append(pr['p'])
        recall.append(pr['r'])
        if i > 0 and i % 10 == 0:
            plt.hist(pr['actions_hist'], bins=range(101), edgecolor='black', linewidth=1)
            plt.xlabel("Number of actions")
            plt.ylabel("Frequency")
            plt.title("Number of Actions Taken On Test Images (Epoch {})".format(i))
            plt.savefig(os.path.join("plots", "hist_epoch_{}.png".format(i)), bbox_inches="tight")
            plt.clf()
    np.savez("precision_and_recall.npz", precision=precision, recall=recall)
    plt.plot(precision)
    plt.plot(recall)
    plt.xlabel("epoch")
    plt.ylabel("precision and recall with IoU threshold 0.50")
    plt.title("Precision and Recall Per Epoch")
    plt.legend(labels = ["Precision", "Recall"])
    plt.savefig(os.path.join("plots", "precision_recall.png"), bbox_inches="tight")
    plt.clf()