from dqn import *
from optimization import *
from dataloader import *
from reinforcement import *
import time, math, random
from itertools import count

# Hyperparameters / utilities
BATCH_SIZE = 128
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.1
EPS_LEN = 6 # number of epochs to decay epsilon
TARGET_UPDATE = 10

eps_sched = np.linspace(EPS_START, EPS_END, EPS_LEN)

# networks
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def select_action(states, eps):
    img_t, action_history = state_transform(states)
    sample = random.random()
    if sample > eps:
        # select best action from model with probability 1-epsilon
        with torch.no_grad():
            actions = policy_net(img_t, action_history)
            actions = torch.max(actions, 1).indices
            print("q-network:", actions)
            return actions
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
        actions = torch.tensor(actions, device=device)
        print("random:", actions)
        return actions

# optimizer / memory
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(10000)

# training data
VOCtrain = torchvision.datasets.VOCDetection("VOC2012", image_set='train')
train_loader = torch.utils.data.DataLoader(VOCtrain, batch_size=TARGET_UPDATE, collate_fn=default_collate)

total_time = 0
for i_epoch in count():
    epoch_start = time.time()
    if i_epoch < EPS_LEN:
        eps = eps_sched[i_epoch]
    else:
        eps = EPS_END
    for i_batch, states in enumerate(train_loader):
        print("Running batch {0} of epoch {1}...".format(i_batch, i_epoch))
        batch_steps = 0
        start = time.time()
        # perform actions on batch items until done
        while len(states) > 0 and batch_steps < 200:
            actions = select_action(states, eps)
            states_new = []
            # store state transition for each each (state, action) pair
            for j in range(actions.shape[0]):
                action = actions[j].item()
                state = states[j]
                reward, next_state, done = take_action(state, action)
                reward = torch.tensor([reward], device=device)
                memory.push(state, action, next_state, reward)
                if not done:
                    states_new.append(next_state)
            # optimize after each action on the batch
            optimize_model(optimizer, memory, policy_net, target_net, BATCH_SIZE, GAMMA)
            batch_steps+=1
            states = states_new
        # update target network after batch
        target_net.load_state_dict(policy_net.state_dict())
       
        stop = time.time()
        t = (stop-start)/60
        print("Finished batch {0} in {1:.2f} minutes.".format(i_batch, t))
    
    t = (time.time() - epoch_start)/60
    print("Finished epoch {0} in {1:.2f} minutes.".format(i_epoch, t))

    total_time += t
    print("Total time: {0:.2f} minutes.".format(total_time))

    # save model after each epoch
    torch.save(target_net, "models/target_net_{}.pt".format(i_epoch))
