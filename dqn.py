import torch.nn as nn
from collections import namedtuple
import torch
import torchvision
import random

# define DQN with resnet preprocessing step

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # pre-trained convolutional network
        conv = torchvision.models.resnet50(pretrained=True)
        modules = list(conv.children())[:-1]
        self.conv = nn.Sequential(*modules)
        for p in conv.parameters():
            p.requires_grad = False
            
        # deep Q-network
        self.dqn = nn.Sequential(
            nn.Linear(2138, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 9),
            nn.Softmax(dim=1)
        )
        
    def forward(self, img_t, action_history):
        out = self.conv(img_t)
        out = out.reshape(out.size(0), 2048)
        out = torch.cat((out, action_history), dim=1)
        out = self.dqn(out)
        return out

# define replay memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
