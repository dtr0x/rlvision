import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from reinforcement import take_action, calculate_iou
from SingleClassDetection import *
import os, sys, time

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
print("Test set size: {}".format(len(VOCtest)))
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
    start = time.time()
    model_path = "models/" + class_name
    recalls = []
    mean_actions = []
    median_actions = []
    n_models = len(os.listdir(model_path))

    for i in range(n_models):
        print("Evaluating model {}...".format(i))
        model = os.path.join(model_path, "target_net_{}.pt".format(i))
        net = torch.load(model).to(device)
        net.eval()
        r, ah = recall(net)
        recalls.append(r)
        mean_actions.append(np.mean(ah))
        median_actions.append(np.median(ah))

    recalls = np.asarray(recalls)
    np.savez("evaluation/eval_{}.npz".format(class_name), 
        recalls=recalls, mean_actions=mean_actions, median_actions=median_actions)

    end = time.time()
    print("Completed evaluation of {0} in {1} seconds.".format(class_name, end-start))

    # plot recall
    plt.plot(recalls)
    plt.xlabel("epoch")
    plt.ylabel("Recall with IoU threshold 0.50")
    plt.title("Recall per epoch for {} data".format(class_name))
    plt.savefig("evaluation/recall_{}.png".format(class_name), bbox_inches="tight")
    plt.clf()

    # plot actions
    plt.plot(mean_actions)
    plt.plot(median_actions)
    plt.xlabel("epoch")
    plt.ylabel("actions")
    plt.title("Number of actions during successful localization of {} class".format(class_name))
    plt.legend(labels = ["mean actions", "median actions"])
    plt.savefig("evaluation/actions_{}.png".format(class_name), bbox_inches="tight")
    plt.clf()