from dqn import *
from optimization import *
from dataloader import *
from reinforcement import *
from visualization import *
from PlaneDetection import PlaneDetection
import time, math, random

# Hyperparameters / utilities
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
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
train_loader = torch.utils.data.DataLoader(VOCtrain, batch_size=1, collate_fn=default_collate)
_, s_init = next(enumerate(train_loader))

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

num_episodes = 50

for i in range(num_episodes):
    print("Running episode {}...".format(i))
    done = False
    action_sequence = []
    s = s_init
    epoch_steps = 0
    while not done and epoch_steps < 40:
        state = s[0]
        img_t, action_history = state_transform(s)
        action = select_action(img_t, action_history, s)[0].item()
        action_sequence.append(action)
        reward, next_state, done = take_action(state, action)
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        optimize_model(optimizer, memory, policy_net, target_net)
        s = [next_state]
        epoch_steps += 1
    print("Action sequence taken during episode {}:".format(i), action_sequence)

    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        action_sequence = localize(s_init[0], "fixed_env/test_{}".format(i), target_net)
        print("Action sequence taken after target update {}:".format(i/TARGET_UPDATE), action_sequence)
