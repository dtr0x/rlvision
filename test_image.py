# create a test visualization
from visualization import *
import torchvision, torch
VOCtest = torchvision.datasets.VOCDetection("VOC2012", image_set='val')
from dataloader import *
test_loader = torch.utils.data.DataLoader(VOCtest, batch_size=1, collate_fn=default_collate, shuffle=True)
test_iter = enumerate(test_loader)
from dqn import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DQN().to(device)
_, [s] = test_iter.__next__()
s = s[0]
localize(s, 'test', net)