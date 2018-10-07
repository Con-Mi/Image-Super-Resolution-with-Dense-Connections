""" ============== Optimization ============ """

import torch
import torch.nn as nn
import torch.optim as optim
from batchnorm_model import densenetSR
from dataLoader import DIV2K_TrainData, DIV2K_ValidData
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()

# Hyperparameters
batch_size = 1
nr_epochs = 10
learning_rate = 0.001
running_loss = 0.0
gamma = 0.1
milestones = [1, 3, 5, 7, 9]

SRmodel = densenetSR()
if use_cuda:
    SRmodel = SRmodel.cuda()

# Training Data and Dataloader
train_data_set = DIV2K_TrainData()
train_dataloader = DataLoader(train_data_set, batch_size = 1, shuffle = True, num_workers = 6)

# Validation Data and Dataloader
valid_data_set = DIV2K_ValidData()
valid_dataloader = DataLoader(valid_data_set, batch_size = 1, shuffle = True, num_workers = 6)

# Optimization
optimizer = optim.ASGD(SRmodel.parameters(), lr = learning_rate)
criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma = gamma)

for epoch in range(nr_epochs):
    for i, sample in enumerate(train_dataloader):
        inputs, labels = sample
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # Zero parameter Gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = SRmodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print data
        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    # Scheduler
    scheduler.step()

torch.save(SRmodel.state_dict(), "SRmodel.pt")
# Load model
# model.load_state_dict(torch.load("SRmodel.pt"))
# model.eval()
 
