
# ________ Augment the data and save the augmentations ________

from dataLoader import DIV2K_TrainData, DIV2K_ValidData, DIV2K_AugmentTrainData
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision import transforms
from tqdm import tqdm

transforms_list = [ transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p = 1.0), 
                    transforms.RandomRotation(205), transforms.RandomRotation(45), transforms.RandomRotation(145), 
                    transforms.RandomRotation(300), transforms.ColorJitter(brightness=1.3), transforms.ColorJitter(contrast=1.2),
                    transforms.ColorJitter(saturation=1.2), transforms.ColorJitter(saturation=0.7), transforms.ColorJitter(hue=0.3),
                    transforms.ColorJitter(hue=0.1) ]

for i, transform_choice in tqdm(enumerate(transforms_list), total=len(transforms_list)):
        train_data_set = DIV2K_AugmentTrainData(data_transforms = transforms.Compose([transform_choice, transforms.ToTensor()]))
        train_dataloader = DataLoader(train_data_set, batch_size = 1, shuffle = True, num_workers = 7)

        for i_idx, sample in enumerate(train_dataloader):
                img_input, label_img = sample
                utils.save_image(img_input, "./DIV2K_train_LR_bicubic/X4/" + str("%04d" % (i_idx + (801 * (i+1)))) + "x4" + ".png", nrow=0, padding=0)
                utils.save_image(label_img, "./DIV2K_train_LR_bicubic/X2/" + str("%04d" % (i_idx + (801 * (i+1)))) + "x2" + ".png", nrow=0, padding=0)
