""" ============== DataLoader ============ """

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import utils
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import numpy as np

class DIV2KDataset(Dataset):
	def __init__(self, file_list_idx, file_list_label_idx, transform = None, mode = "train"):
		self.data_root = "./"
		self.file_list_idx = file_list_idx
		self.file_list_label_idx = file_list_label_idx
		self.transform = transform
		self.mode = mode
		# NOTE: This takes care only the downscaling 2 dataset.
		if self.mode is "train":
			self.data_dir = os.path.join(self.data_root, "DIV2K_train_LR_bicubic")
			self.label_dir = os.path.join(self.data_root, "DIV2K_train_LR_bicubic")
		elif self.mode is "validation":
			self.data_dir = os.path.join(self.data_root, "DIV2K_valid_LR_bicubic")
			self.label_dir = os.path.join(self.data_root, "DIV2K_valid_LR_bicubic")

	def __len__(self):
		return len(self.file_list_idx)

	def __getitem__(self, index):
		if index not in range(len(self.file_list_idx)):
			return self.__getitem__(np.random.randint(0, self.__len__()))
		
		file_id_image = self.file_list_idx["ids"].iloc[index]
		file_id_label = self.file_list_label_idx["ids"].iloc[index]

		if self.mode is "train":
			self.image_folder = os.path.join(self.data_dir, "X4")
			self.label_folder = os.path.join(self.label_dir, "X2")
			self.image_path = os.path.join(self.image_folder, file_id_image)
			self.label_path = os.path.join(self.label_folder, file_id_label)
			image = Image.open(self.image_path)
			label = Image.open(self.label_path)
			if self.transform is not None:
				image = self.transform(image)
				label = self.transform(label)
			return image,label

		elif self.mode is "validation":
			self.image_folder = os.path.join(self.data_dir, "X4")
			self.image_path = os.path.join(self.image_folder, file_id_image)
			self.label_folder = os.path.join(self.label_dir, "X2")
			self.label_path = os.path.join(self.label_folder, file_id_label)
			image = Image.open(self.image_path)
			label = Image.open(self.label_path)
			if self.transform is not None:
				image = self.transform(image)
				label = self.transform(label)
			return image, label

def DIV2K_AugmentTrainData(data_transforms = None):
	file_list = pd.read_csv("train_data_index.csv")
	data_set = DIV2KDataset(file_list, transform = data_transforms)
	return data_set

def DIV2K_TrainData(**kwargs):
	file_list = pd.read_csv("train_data_index_x4.csv")
	file_list_labels = pd.read_csv("train_data_index_x2.csv")
	data_transforms = transforms.Compose([transforms.ToTensor()])
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	data_set = DIV2KDataset(file_list, file_list_labels, transform = data_transforms)
	return data_set

def DIV2K_ValidData(**kwargs):
	file_list = pd.read_csv("valid_data_index_x4.csv")
	file_list_labels = pd.read_csv("valid_data_index_x2.csv")
	data_transforms = transforms.Compose([transforms.ToTensor()])
	data_set = DIV2KDataset(file_list, file_list_labels, transform = data_transforms, mode = "validation")
	return data_set
