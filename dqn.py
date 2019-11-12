import torch
import torch.nn as nn
import torchvision

# DQN with resnet preprocessing step
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # pre-trained convolutional network
        conv = torchvision.models.resnet50(pretrained=True)
        # remove the last last layer to extract features
        modules = list(conv.children())[:-1]
        self.conv = nn.Sequential(*modules)
        # don't perform gradient descent through the classifier 
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
        
    # forward pass through the network    
    def forward(self, img_t, action_history):
        out = self.conv(img_t)
        out = out.reshape(out.size(0), 2048)
        out = torch.cat((out, action_history), dim=1)
        out = self.dqn(out)
        return out
