""" ============== Optimization ============ """

import torch
import torch.nn as nn
import torch.optim as optim
from model import densenetSR
from dataLoader import DIV2K_TrainData, DIV2K_ValidData
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()

# Hyperparameters
batch_size = 1
nr_epochs = 20
learning_rate = 0.001

SRmodel = densenetSR()
if use_cuda:
    SRmodel = SRmodel.cuda()

# Training Data and Dataloader
train_data_set = DIV2K_TrainData()
train_dataloader = DataLoader(train_data_set, batch_size = 1, shuffle = True, num_workers = 6)

# Validation Data and Dataloader
valid_data_set = DIV2K_ValidData()
valid_dataloader = DataLoader(valid_data_set, batch_size = 1, shuffle = True, num_workers = 6)


optimizer = optim.ASGD(SRmodel.parameters(), lr = learning_rate)
criterion = nn.SmoothL1Loss()

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
        if i % 200 == 199:
            print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

