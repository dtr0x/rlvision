import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

DEFAULT_MODEL_PATH = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet(torch.nn.Module):
  
    def __init__(self, model_path=DEFAULT_MODEL_PATH, *args, **kwargs):
      
        # Call to super-constructor
        super(ResNet, self).__init__()
        self.path = model_path
        # Sets up our layers
        self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_4 = torch.nn.Conv2d(64, 64, 3, padding=1)
                
        torch.nn.init.xavier_uniform_(self.conv2_1.weight)
        torch.nn.init.xavier_uniform_(self.conv2_2.weight)
        torch.nn.init.xavier_uniform_(self.conv2_3.weight)
        torch.nn.init.xavier_uniform_(self.conv2_4.weight)
        
        self.conv3_b = torch.nn.Conv2d(64, 128, 1, stride=2)
        self.conv3_1 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_4 = torch.nn.Conv2d(128, 128, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3_b.weight)
        torch.nn.init.xavier_uniform_(self.conv3_1.weight)
        torch.nn.init.xavier_uniform_(self.conv3_2.weight)
        torch.nn.init.xavier_uniform_(self.conv3_3.weight)
        torch.nn.init.xavier_uniform_(self.conv3_4.weight)

        self.conv4_b = torch.nn.Conv2d(128, 256, 1, stride=2)
        self.conv4_1 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv4_b.weight)
        torch.nn.init.xavier_uniform_(self.conv4_1.weight)
        torch.nn.init.xavier_uniform_(self.conv4_2.weight)
        torch.nn.init.xavier_uniform_(self.conv4_3.weight)
        torch.nn.init.xavier_uniform_(self.conv4_4.weight)        

        self.conv5_b = torch.nn.Conv2d(256, 512, 1, stride=2)
        self.conv5_1 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_4 = torch.nn.Conv2d(512, 512, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv5_b.weight)
        torch.nn.init.xavier_uniform_(self.conv5_1.weight)
        torch.nn.init.xavier_uniform_(self.conv5_2.weight)
        torch.nn.init.xavier_uniform_(self.conv5_3.weight)
        torch.nn.init.xavier_uniform_(self.conv5_4.weight)
        
        self.pool_avg = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = torch.nn.Linear(512, 2)

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv layer2
        x_bypass = x
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x) + x_bypass
        x = F.relu(x)
        x = self.conv2_3(x)
        x = F.relu(x)
        x = self.conv2_4(x) + x_bypass
        x = F.relu(x)
        
        # Conv layer3
        
        x_bypass = self.conv3_b(x)
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x) + x_bypass
        x = F.relu(x)
        
        x_bypass = x
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.conv3_4(x) + x_bypass
        x = F.relu(x)
        
        # Conv layer4
        x_bypass = self.conv4_b(x)
        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x) + x_bypass
        x = F.relu(x)
        
        x_bypass = x
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.conv4_4(x) + x_bypass
        x = F.relu(x)
        
        #Conv layer5
        
        x_bypass = self.conv5_b(x)
        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x) + x_bypass
        x = F.relu(x)
        
        x_bypass = x
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.conv5_4(x) + x_bypass
        x = F.relu(x)
        
        # Average pool
        x = self.pool_avg(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = F.softmax(x)

        return x

    def save(self, path=None):
        """Saves the model to the desired path."""
        if path is None:
            path = self.path
            
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        """Loads the model from desired path."""
        if path is None:
            path = self.path
            
        self.load_state_dict(torch.load(path))

    def train_model(self, train_loader, valid_loader,
                  num_epochs=10, save_mode='every',
                    smart_detection=True, silent=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
                
        criterion = torch.nn.CrossEntropyLoss()
        validation_loss_per_epoch = []
        training_loss_per_epoch = []
        training_error_per_epoch = []
        valid_error_per_epoch = []
        print('Training...')
        for epoch in range(num_epochs):
            training_loss = []
            correct_trainning = 0
            total = 0
            correct = 0
            self.train()
            for i, data in enumerate(train_loader):
                inputs, target = data
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                input_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)
                output = self(input_var)
                loss = criterion(output, target_var)
                reg = 1e-6
                l2_loss = None
                for name, param in self.named_parameters():
                    if 'bias' not in name: 
                        if l2_loss is None:
                            l2_loss = 0.5 * reg *torch.sum(torch.pow(param, 2))
                        else:
                            l2_loss += (0.5 * reg *torch.sum(torch.pow(param, 2)))

                loss += l2_loss
                training_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += target_var.size(0)
                correct += (predicted == target_var).sum().item()
                if i % 50 == 0:
                    print("Done batch {}. Loss: {}. Correct: {}".format(i, loss.item(), (predicted == target_var).sum().item()))
            training_loss_per_epoch.append(np.mean(training_loss))
            training_error_per_epoch.append(1 - (correct/total))
            print(correct/total)
            validation_loss = [];
            val_size = len(list(valid_loader))
            val_correct = 0 
            val_total = 0
            self.eval()
            
            for i, data in enumerate(valid_loader, 0):
                inputs,target = data
                inputs = inputs.to(device)
                target = target.to(device)
                input_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)
                output = self(input_var)
                loss = criterion(output, target_var)
                validation_loss.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                
                val_total += target_var.size(0)
                val_correct += (predicted == target_var).sum().item()
                if i % 50 == 0:
                    print("Done batch validation {}, Loss: {}, Correct: {}".format(i, loss.item(), (predicted == target_var).sum().item()))
            validation_loss_per_epoch.append(np.mean(validation_loss))
            validation_accuracy = val_correct/val_total
            print("Validation Accuracy: {}".format(validation_accuracy))
            valid_error_per_epoch.append(1 - validation_accuracy)
            print("done epoch ", epoch)

            if save_mode == 'every' or epoch % save_mode == 0:
                self.save(os.path.join(self.path, "net_{}.pth".format(epoch)))

        return training_loss_per_epoch,validation_loss_per_epoch,training_error_per_epoch,valid_error_per_epoch

    def compute_layer_size(self, input_size, kernel_size, padding = 0, stride = 1, dilation = 1):
    
        ks = (kernel_size) + (dilation - 1) * (kernel_size - 1)
        
        return np.floor((input_size - ks - (2 * padding)) / (stride)) - 1