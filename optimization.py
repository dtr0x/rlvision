import torch
import torch.nn.functional as F
from collections import namedtuple
from dataloader import state_transform
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_last_action(state):
    last_action = state.action_history[:9]
    return last_action.nonzero().item()

def optimize_model(optimizer, memory, policy_net, target_net, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: get_last_action(s) != 8,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = [s for s in batch.next_state if get_last_action(s) != 8]
    non_final_img_t, non_final_action_history = state_transform(non_final_next_states)
    
    state_batch = batch.state
    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.cat(batch.reward)
       
    img_t, action_history = state_transform(state_batch)
    
    actions = policy_net(img_t, action_history)

    state_action_values = policy_net(img_t, action_history).gather(1, action_batch.view(-1, 1))

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_img_t, non_final_action_history).max(1)[0].detach()

    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp(-1, 1)
    optimizer.step()
