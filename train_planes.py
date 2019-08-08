from dqn import *
from optimization import *
from dataloader import *
from reinforcement import *
from visualization import *
from PlaneDetection import PlaneDetection
import time, math, random

# Hyperparameters / utilities
BATCH_SIZE = 30
NUM_EPOCHS = 50
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# networks
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer / memory
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(10000)

# datasets
VOCtrain = PlaneDetection('train')
VOCtest = PlaneDetection('val')
train_loader = torch.utils.data.DataLoader(VOCtrain, batch_size=BATCH_SIZE, collate_fn=default_collate)
test_loader = torch.utils.data.DataLoader(VOCtest, batch_size=1, collate_fn=default_collate)
test_iter = enumerate(test_loader)

_, [s_fixed] = test_iter.__next__()

steps_done = 0

def select_action(img_t, action_history, states):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # select best action from model with probability 1-epsilon
        with torch.no_grad():
            actions = policy_net(img_t, action_history)
            return torch.max(actions, 1).indices
    else:
        # select random positive action with probability epsilon
        actions = []
        for state in states:
            positive_actions = find_positive_actions(state)
            if len(positive_actions) > 0:
                action = random.choice(positive_actions)
            else:
                action = random.randrange(9)
            actions.append(action)
        return torch.tensor(actions, device=device)

total_time = 0
print("First 10 DQN params (initialization):", policy_net.state_dict()['dqn.0.weight'][0][:10])
for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    for i, states in enumerate(train_loader):
        print("Running batch", i)
        batch_steps = 0
        start = time.time()
        while len(states) > 0 and batch_steps < 100:
            img_t, action_history = state_transform(states)
            actions = select_action(img_t, action_history, states)
            states_new = []
            for j in range(actions.shape[0]):
                action = actions[j].item()
                state = states[j]
                reward, next_state, done = take_action(state, action)
                reward = torch.tensor([reward], device=device)
                memory.push(state, action, next_state, reward)
                if not done:
                    states_new.append(next_state)
    
            optimize_model(optimizer, memory, policy_net, target_net, BATCH_SIZE, GAMMA)
            
            if batch_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            states = states_new
            batch_steps+=1
        
        # save visualization
        _, [s] = test_iter.__next__()
        localize(s_fixed, "fixed/img_{0}-{1}".format(epoch, i), target_net)
        localize(s, "img_{0}-{1}".format(epoch, i), target_net)
        
        stop = time.time()
        t = (stop-start)/60
        total_time += t
        print("Finished batch {0} in {1:.2f} minutes.".format(i, t))
        print("Total time: {0:.2f} minutes.".format(total_time))
        print("First 10 DQN params after batch {0}:".format(i), policy_net.state_dict()['dqn.0.weight'][0][:10])
    
    t = (time.time() - epoch_start)/60
    print("Finished epoch {0} in {1:.2f} minutes.".format(epoch, t))
    torch.save(target_net, "models/target_net_{}.pt".format(epoch))