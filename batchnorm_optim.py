""" ============== Optimization ============ """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from batchnorm_model import densenetSR
from dataLoader import DIV2K_TrainData, DIV2K_ValidData

import time
import copy
from tqdm import tqdm

use_cuda = torch.cuda.is_available()

# Hyperparameters
batch_size = 1
nr_epochs = 10
momentum = 0.93
learning_rate = 0.01
running_loss = 0.0
gamma = 0.1
milestones = [1, 3, 5, 7, 9]

SRmodel = densenetSR()
if use_cuda:
    SRmodel = SRmodel.cuda()

# Training Data and Dataloader
train_data_set = DIV2K_TrainData()
train_dataloader = DataLoader(train_data_set, batch_size = 1, shuffle = True, num_workers = 20)

# Validation Data and Dataloader
valid_data_set = DIV2K_ValidData()
valid_dataloader = DataLoader(valid_data_set, batch_size = 1, shuffle = True, num_workers = 20)

# Optimization
optimizer = optim.SGD(SRmodel.parameters(), lr = learning_rate, momentum = momentum)
criterion = nn.SmoothL1Loss().cuda() if use_cuda else nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma = gamma)

dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader}

def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs=10, scheduler = None):
    start_time = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(cust_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("_"*15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            elif phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            running_corrects = 0

            for input_img, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                input_img = input_img.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs = cust_model(input_img)
                    loss = criterion(outputs, labels)
                    #_, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])     # Different uses for this (len(dataloaders[phase].dataset)) 
            # epoch_acc = running_corrects.double() / len(dataloaders[phase])
            epoch_acc = 0.0

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if scheduler is not None and phase == "train":
                scheduler.step()
            

            if phase == 'valid' and epoch_acc > best_acc:
                # best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(cust_model.state_dict)
                pass
            if phase == "valid":
                # val_acc_history.append(epoch_acc)
                pass
        
        print()
    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation Accuracy: {:4f}".format(best_acc))
    
    best_model_wts = copy.deepcopy(cust_model.state_dict())
    cust_model.load_state_dict(best_model_wts)
    return cust_model, val_acc_history


def save_model(cust_model, name = "SRmodel.pt"):
    torch.save(cust_model.state_dict(), name)

def load_model(cust_model, model_dir = "./SRmodel.pt"):
    cust_model.load_state_dict(torch.load(model_dir))
    cust_model.eval()
    return cust_model

print("Start Training for 2 epochs: ")
print("*"*15)
SRmodel, validation_acc = train_model(SRmodel, dataloaders_dict, criterion, optimizer, nr_epochs, scheduler)
save_model(SRmodel, name = "SRmodel_selu.pt")
