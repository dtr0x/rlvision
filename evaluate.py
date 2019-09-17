import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from reinforcement import take_action, calculate_iou
from SingleClassDetection import *
import os, sys

voc_classes = {}
f = open("voc_classes.txt", "r")
lines = f.readlines()
for l in lines:
    k,v = l.split(',')
    voc_classes[k] = int(v)

try:
    class_name = sys.argv[1]
    if class_name not in voc_classes.keys():
        raise IndexError()
except IndexError:
    print("Must provide class name from one of:")
    print("\n".join(voc_classes.keys()))
    exit()

# test data
VOCtest = SingleClassDetection(class_name, 'val')
test_loader = torch.utils.data.DataLoader(VOCtest, 
    batch_size=1, collate_fn=default_collate)

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

def recall(net):
    tp = 0
    actions_hist = []
    for i, [s] in enumerate(test_loader):
        n_actions, iou = localize(s, net)
        if iou is not None and iou >= 0.5:
            tp += 1
            actions_hist.append(n_actions)
    recall = tp/len(VOCtest) # all training examples have a detectable object
    return recall, actions_hist

if __name__ == "__main__":
    model_path = "models/" + class_name
    recalls = []
    mean_actions = []
    n_models = len(os.listdir(model_path))
    for i in range(n_models):
        print("Evaluating model {}...".format(i))
        model = os.path.join(model_path, "target_net_{}.pt".format(i))
        net = torch.load(model).to(device)
        net.eval()
        r, ah = recall(net)
        recalls.append(r)
        mean_actions.append(np.mean(ah))
    recalls = np.asarray(recalls)
    np.savez("recall_{}.npz".format(class_name), recalls=recalls, mean_actions=mean_actions)
    plt.plot(recall)
    plt.xlabel("epoch")
    plt.ylabel("Recall with IoU threshold 0.50")
    plt.title("Recall per epoch for {} data".format(class_name))
    plt.savefig(os.path.join("plots", "recall_{}.png".format(class_name)), bbox_inches="tight")
    plt.clf()