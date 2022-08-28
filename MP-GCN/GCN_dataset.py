import numpy as np
import h5py
from torch.utils.data import Dataset

dist_file = 'path/to/distances/file.hdf'

''' Define a class to contain the data that will be included in the dataloader 
sent to the MP-GCN model '''
class GCN_Dataset(Dataset):
  
    def __init__(self, data_file, output_info=False, cache_data=True):
        super(GCN_Dataset, self).__init__()
        self.data_file = data_file
        self.output_info = output_info
        self.cache_data = cache_data
        self.data_dict = {}  # used to store data
        self.data_list = []  # used to store id's for data
        
        # retrieve PDB id's and affinities from h5py file
        with h5py.File(data_file, 'r') as f:
            for pdbid in f.keys():
                affinity = np.asarray(f[pdbid].attrs['affinity']).reshape(1, -1)
                self.data_list.append((pdbid, affinity))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.cache_data:
            if item in self.data_dict.keys():
                return self.data_dict[item]
            else:
                pass       
        pdbid, affinity = self.data_list[item]
        node_feats, coords = None, None

        dist_hdf = h5py.File(dist_file, 'r')
        dists = np.array(dist_hdf[pdbid])
        dist_hdf.close()
        
        if self.cache_data:
            if self.output_info:
                self.data_dict[item] = (pdbid, dists)
            else:
                self.data_dict[item] = dists
            return self.data_dict[item]
        else:
            if self.output_info:
                return (pdbid, dists)
            else:
                return dists
