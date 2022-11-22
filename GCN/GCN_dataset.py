import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances

''' Define a class to contain the data that will be included in the dataloader 
sent to the MP-GCN model '''

class GCN_Dataset(Dataset):
  
    def __init__(self, data_file):
        super(GCN_Dataset, self).__init__()
        self.data_file = data_file
        self.data_dict = {}
        self.data_list = []
        
        # retrieve PDB id's and affinities from hdf file
        with h5py.File(data_file, 'r') as f:
            for pdbid in f.keys():
                affinity = np.asarray(f[pdbid].attrs['affinity']).reshape(1, -1)
                self.data_list.append((pdbid, affinity))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        if item in self.data_dict.keys():
            return self.data_dict[item]

        pdbid, affinity = self.data_list[item]
        node_feats, coords = None, None

        coords=h5py.File(self.data_file,'r')[pdbid][:,0:3]
        dists=pairwise_distances(coords, metric='euclidean')
        
        self.data_dict[item] = (pdbid, dists)
        return self.data_dict[item]
