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
    model_path = "plane_models"
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
    mean_actions[np.isnan(mean_actions)] = np.inf
    recalls = np.concatenate((np.zeros(4), recalls))
    np.savez("recall_planes.npz", recalls=recalls, mean_actions=mean_actions)
    #plt.plot(recall)
    #plt.xlabel("epoch")
    #plt.ylabel("Recall with IoU threshold 0.50")
    #plt.title("Precision and Recall Per Epoch")
    #plt.legend(labels = ["Precision", "Recall"])
    #plt.savefig(os.path.join("plots", "precision_recall.png"), bbox_inches="tight")
    #plt.clf()