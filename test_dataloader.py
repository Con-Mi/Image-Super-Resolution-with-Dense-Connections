### Test the Dataloader

from dataLoader import DIV2K_TrainData, DIV2K_ValidData
from torch.utils.data import DataLoader

# Training Data and Dataloader
train_data_set = DIV2K_TrainData()
train_dataloader = DataLoader(train_data_set, batch_size = 1, shuffle = True, num_workers = 6)

for i, sample in enumerate(train_dataloader):
        inputs, labels = sample
        print(inputs.size)
        print(i)