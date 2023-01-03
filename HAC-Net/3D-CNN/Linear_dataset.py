import numpy as np
import torch
from torch.utils.data import Dataset

""" Define a class to contain the extracted 3D-CNN features that will be included in the dataloader 
sent to the fully-connected network """

class Linear_Dataset(Dataset):
	def __init__(self, npy_path, feat_dim=22):
		super(Linear_Dataset, self).__init__()
		self.npy_path = npy_path
		self.input_feat_array = np.load(npy_path, allow_pickle=True)[:,:-1].astype(np.float32)
		self.input_affinity_array = np.load(npy_path, allow_pickle=True)[:,-1].astype(np.float32)
		self.data_info_list = []
		
	def __len__(self):
		count = self.input_feat_array.shape[0]
		return count

	def __getitem__(self, idx):
		data, affinity = self.input_feat_array[idx], self.input_affinity_array[idx]

		x = torch.tensor(data)
		y = torch.tensor(np.expand_dims(affinity, axis=0))
		return x,y
