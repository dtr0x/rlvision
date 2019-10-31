import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import csv
import pandas as pd
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from IPython.display import Image
from ResNet import *
to_img = ToPILImage()

BATCH_SIZE = 32

train_dir = "data/trainset"
valid_dir = "data/validset"

transform =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_datasets = datasets.ImageFolder(train_dir, transform)
valid_datasets = datasets.ImageFolder(valid_dir, transform)

class_names = train_datasets.classes

train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, pin_memory=True)

validation_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, pin_memory=True)

n_epochs = 100
cnn = ResNet().to(device)
print(device)
training_loss_per_epoch,validation_loss_per_epoch,training_error_per_epoch,valid_error_per_epoch = cnn.train_model(train_dataloaders, validation_dataloaders,num_epochs = n_epochs)
    
epoch_range = list(range(n_epochs))
plt.plot(epoch_range, training_loss_per_epoch, 'g^', epoch_range, validation_loss_per_epoch, 'bs')
plt.ylabel("loss")
plt.xlabel("epoch iteration")
plt.savefig("plots/loss.png", bbox_inches="tight")
plt.clf()

plt.plot(epoch_range, training_error_per_epoch, 'g^', epoch_range, valid_error_per_epoch, 'bs')
plt.ylabel("error")
plt.xlabel("epoch iteration")
plt.savefig("plots/error.png", bbox_inches="tight")
plt.clf() 

print("Best epoch by validation error: {}".format(np.argmin(validation_loss_per_epoch)))
