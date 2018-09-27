""" ============== DataLoader ============ """

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import pandas as pd
from PIL import Image

class DIV2KDataset(Dataset):
	def __init__(self, file_list_idx, transform = None, mode = "train"):
		self.data_root = "../DIV2K"
		self.file_list_idx = file_list_idx
		self.transform = transform
		self.mode = mode
		# NOTE: This takes care only the downscaling 2 dataset.
		if self.mode is "train":
			self.data_dir = os.path.join(self.data_root, "DIV2K_train_LR4_bicubic")
			self.label_dir = os.path.join(self.data_root, "DIV2K_train_LR2_bicubic")
		elif self.mode is "validation":
			self.data_dir = os.path.join(self.data_root, "DIV2K_valid_LR4_bicubic")
			self.label_dir = os.path.join(self.data_root, "DIV2K_valid_LR2_bicubic")

	def __len__(self):
		return len(self.file_list_idx)

	def __getitem__(self, index):
		if index not in range(len(self.file_list_idx)):
			return self.__getitem__(np.random.randint(0, self.__len__()))
		
		file_id = self.file_list_idx.iloc[index]

		if self.mode is "train":
			image_folder = os.path.join(self.data_dir, "X4")
			label_folder = os.path.join(self.label_dir, "X2")
			image_path = os.path.join(image_folder, str("%04d" % file_id) + "x4" + ".png")
			label_path = os.path.join(self.label_folder, str("%04d" % file_id) + "x2" + ".png")
			image = Image.open(image_path)
			label = Image.open(label_path)
			if self.transform is not None:
				image = self.transform(image)
				label = self.transform(label)
			return image,label

		elif self.mode is "validation":
			image_folder = os.path.join(self.data_dir, "X4")
			image_path = os.path.join(image_folder, str("%04d" % file_id) + "x4" + ".png") # Need to get images in the {0001, 0010, ..} format
			label_folder = os.path.join(self.label_dir, "X2")
			label_path = os.path.join(self.label_folder, str("%04d" % file_id) + "x2" + ".png")
			image = Image.open(image_path)
			label = Image.open(label_path)
			if self.transform is not None:
				image = self.transform(image)
				label = self.transform(label)
			return image, label

file_list = pd.read_csv("train_data_index.csv")
data_transforms = transforms.Compose([transforms.ToTensor()])
# data_set = DIV2KDataset(file_list, transform = data_transforms)

#print(data_set)

def DIV2KData(**kwargs):
	data_set = DIV2KDataset(file_list, transform = data_transforms)
	return data_set

"""
for i, sample in enumerate(dataloader):
	image2, label2 = sample
	print(image2.size())
	print(label2.size())
	print(i)
"""
