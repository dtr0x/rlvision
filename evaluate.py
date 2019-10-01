import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from reinforcement import *
from SingleClassDetection import *
from visualization import draw_localization_actions
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
    done = False
    for i in range(100):
        img_t, action_history = state_transform([state])
        action = net(img_t, action_history).max(1).indices[0].item()
        reward, state, done = take_action(state, action)
        n_actions += 1 
        if done:
            break
    conf = calculate_conf(state)
    return n_actions, conf

def recall(net):
    tp = 0
    actions_hist = []
    for i, [s] in enumerate(test_loader):
        n_actions, conf = localize(s, net)
        if conf >= 0.8:
            tp += 1
            actions_hist.append(n_actions)
    recall = tp/len(VOCtest) # all training examples have a detectable object
    return recall, actions_hist

if __name__ == "__main__":
    start = time.time()
    model_path = "models/" + class_name
    save_path = "evaluation/" + class_name
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
    np.savez(os.path.join(save_path, "data"), 
        recalls=recalls, mean_actions=mean_actions, median_actions=median_actions)

    end = time.time()
    print("Completed evaluation of {0} in {1} seconds.".format(class_name, end-start))

    # plot recall
    plt.plot(recalls)
    plt.xlabel("epoch")
    plt.ylabel("Recall with confidence threshold 0.80")
    plt.title("Recall per epoch for {} data".format(class_name))
    plt.savefig(os.path.join(save_path, "recall.png"), bbox_inches="tight")
    plt.clf()

    # plot actions
    plt.plot(mean_actions)
    plt.plot(median_actions)
    plt.xlabel("epoch")
    plt.ylabel("actions")
    plt.title("Number of actions during successful localization of {} class".format(class_name))
    plt.legend(labels = ["mean actions", "median actions"])
    plt.savefig(os.path.join(save_path, "actions.png"), bbox_inches="tight")
    plt.clf()

    # load best model and localize test images
    best_model_idx = np.argmax(recalls)
    best_model = torch.load(os.path.join(model_path, 
        "target_net_{}.pt".format(best_model_idx))).to(device)
    
    success_path = os.path.join(save_path, "success")
    failure_path = os.path.join(save_path, "failure")
    
    max_actions = 100
    n_success_actions = []
    
    for i, [s] in enumerate(test_loader):
        vis, action_sequence, conf = draw_localization_actions(s, max_actions, best_model)
        actions_taken = len(action_sequence)
        if actions_taken == max_actions:
            print("Could not localize item {}.".format(i))
            vis.save(os.path.join(failure_path, "{}.png".format(i)))
        if conf < 0.8:
            print("Localization for item {} failed with confidence < 0.8.".format(i))
            vis.save(os.path.join(failure_path, "{}.png".format(i)))
        else:
            vis.save(os.path.join(success_path, "{}.png".format(i)))
            print("Localized item {} in {} actions.".format(i, actions_taken))
            n_success_actions.append(actions_taken)
    
    print("Successfully localized {0} of {1} items with {2:.2f} average number of actions taken."
        .format(len(n_success_actions), len(VOCtest), np.mean(n_success_actions)))
    
    # best model actions histogram
    plt.hist(n_success_actions, bins=range(max_actions), edgecolor='black', linewidth=1)
    plt.savefig(os.path.join(save_path, "best_model_actions_hist.png"))
    plt.xlabel("Number of actions")
    plt.ylabel("Frequency")
    plt.text(0.4, 0.75, 
        "{0} of{1} items successfully localized\nwith {2:.2f} average number of actions"
        .format(len(n_success_actions), len(VOCtest), np.mean(n_success_actions)),
        transform=plt.gca().transAxes)
    plt.title("Number of Actions Taken On Test Images (Epoch {})".format(best_model_idx))
    plt.savefig(os.path.join(save_path, "best_model_actions_hist.png"))
