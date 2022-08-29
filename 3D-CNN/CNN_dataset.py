import torch
from torch.utils.data import Dataset

''' Define a class to contain the data that will be included in the dataloader 
sent to the 3D-CNN '''
class Dataset_hdf(Dataset):

	def __init__(self, hdf_path, feat_dim=22):
		super(Dataset_hdf, self).__init__()
		self.hdf_path = hdf_path
		self.feat_dim = feat_dim
		self.hdf = h5py.File(self.hdf_path, 'r')
		self.data_info_list = []
    # append PDB id and affinity label to data_info_list
		for pdbid in self.hdf.keys():
			affinity = float(self.hdf[pdbid].attrs['affinity'])
			self.data_info_list.append([pdbid, affinity])

	def close(self):
		self.hdf.close()

	def __len__(self):
		count = len(self.data_info_list)
		return count
		
	def get_data_info_list(self):
		return self.data_info_list

	def __getitem__(self, idx):
		pdbid, affinity = self.data_info_list[idx]

		data = self.hdf[pdbid][:]
		x = torch.tensor(data)
		x = x.permute(3,0,1,2)
		y = torch.tensor(np.expand_dims(affinity, axis=0))
		return x,y, pdbid
