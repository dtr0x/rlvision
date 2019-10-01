import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from reinforcement import *
from SingleClassDetection import *
from visualization import *
import os, sys, time

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
            print("Localized item {} in {} actions.".format(i, n_actions))
    recall = tp/len(VOCtest) # all training examples have a detectable object
    return recall, actions_hist

if __name__ == "__main__":
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
    print("loaded test data.")
    start = time.time()
    
    model_path = "models/" + class_name
    save_path = "evaluation/" + class_name
    recalls = []
    mean_actions = []
    median_actions = []
    #n_models = len(os.listdir(model_path))
    #i = n_models - 1
    i = 299

    print("Evaluating model {}...".format(i))
    model = os.path.join(model_path, "target_net_{}.pt".format(i))
    net = torch.load(model).to(device)
    net.eval()
    recall, ah = recall(net)
    mean_actions = np.mean(ah)
    median_actions = np.median(ah)

    end = time.time()
    print("Completed evaluation of {0} in {1} seconds.".format(class_name, end-start))
    
    print("Successfully localized {0} of {1} items with {2:.2f} average number of actions taken."
        .format(len(ah), len(VOCtest), mean_actions))
    
    # best model actions histogram
    plt.hist(ah, bins=max(ah), edgecolor='black', linewidth=1)
    plt.xlabel("Number of actions")
    plt.ylabel("Frequency")
    plt.text(0.4, 0.75, 
        "{0} of {1} items successfully localized\nwith {2:.2f} average number of actions"
        .format(len(ah), len(VOCtest), mean_actions),
        transform=plt.gca().transAxes)
    plt.title("Number of Actions Taken On Test Images (Epoch {})".format(i))
    plt.savefig(os.path.join(save_path, "actions_hist.png"))
    plt.clf()


    # produce visualization output

    classifier = torchvision.models.resnet50(pretrained=True).to(device)
    classifier.eval()

    # test data
    VOCtest = SingleClassDetection(class_name, 'val', 0.25)
    print("Test set size: {}".format(len(VOCtest)))
    test_loader = torch.utils.data.DataLoader(VOCtest, 
        batch_size=1, collate_fn=default_collate)
    print("loaded test data.")
    success_path = os.path.join(save_path, "success")
    failure_path = os.path.join(save_path, "failure")
    
    max_actions = 100
    n_success_actions = []
    success_states = []
    
    for i, [s] in enumerate(test_loader):
        vis, action_sequence, iou = draw_sequence_with_conf_score(s, max_actions, net, classifier)
        actions_taken = len(action_sequence)
        if actions_taken == max_actions:
            print("Could not localize item {}.".format(i))
            vis.save(os.path.join(failure_path, "{}.png".format(i)))
        if iou < 0.5:
            print("Localization for item {} failed with IOU < 0.5.".format(i))
            vis.save(os.path.join(failure_path, "{}.png".format(i)))
        else:
            vis.save(os.path.join(success_path, "{}.png".format(i)))
            print("Localized item {} in {} actions.".format(i, actions_taken))
            n_success_actions.append(actions_taken)
            success_states.append(s)
    
    print("Successfully localized {0} of {1} items with {2:.2f} average number of actions taken."
        .format(len(n_success_actions), len(VOCtest), np.mean(n_success_actions)))
    