import h5py
import numpy as np
import math
import torch
from torch import nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import DataListLoader, Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp
from scipy.stats import *
from sklearn.metrics import *

def predict(test, cnn_test_path, gcn0_test_path, gcn1_test_path, cnn_checkpoint_path, gcn0_checkpoint_path, gcn1_checkpoint_path):

    """
    Define a function to test the hybrid model, 3D-CNN, or MP-GCN
    Inputs:
    1) test: either "hybrid", "cnn", "gcn0", or "gcn1"    
    2) cnn_test_path: path to cnn test set npy file
    3) gcn0_test_path: path to gcn0 test set hdf file
    4) gcn1_test_path: path to gcn1 test set hdf file
    5) cnn_checkpoint_path: path to cnn checkpoint file
    6) gcn0_checkpoint_path: path to gcn0 checkpoint file
    7) gcn1_checkpoint_path: path to gcn1 checkpoint file
    Output:
    1) Print statement of summary of evaluation: RMSE, MAE, r^2, Pearson r, Spearman p, Mean, SD
    2) Correlation scatter plot of true vs predicted values
    """

    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    device_name = "cuda:0"
    if use_cuda:
        device = torch.device(device_name)
        torch.cuda.set_device(int(device_name.split(':')[1]))
    else:   
        device = torch.device('cpu')  

    if test == 'hybrid' or 'cnn':
        # 3D-CNN
        batch_size = 50
        # load dataset
        cnn_dataset = MLP_Dataset(cnn_test_path)
        # initialize data loader
        batch_count = len(cnn_dataset) // batch_size
        cnn_dataloader = DataLoader(cnn_dataset, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=None)
        # define model
        cnn_model = MLP(use_cuda=use_cuda)
        cnn_model.to(device)
        if isinstance(cnn_model, (DistributedDataParallel, DataParallel)):
            cnn_model = cnn_model.module
        # load checkpoint file
        cnn_checkpoint = torch.load(cnn_checkpoint_path, map_location=device)
        # model state dict
        cnn_model_state_dict = cnn_checkpoint.pop("model_state_dict")
        cnn_model.load_state_dict(cnn_model_state_dict, strict=False)
        # create empty arrays to hold predicted and true values
        y_true_cnn = np.zeros((len(cnn_dataset),), dtype=np.float32)
        y_pred_cnn = np.zeros((len(cnn_dataset),), dtype=np.float32)
        pdbid_arr = np.zeros((len(cnn_dataset),), dtype=object)
        pred_list = []
        cnn_model.eval()
        with torch.no_grad():
            for batch_ind, batch in enumerate(cnn_dataloader):
                x_batch_cpu, y_batch_cpu = batch
                x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
                bsize = x_batch.shape[0]
                ypred_batch, _ = cnn_model(x_batch[:x_batch.shape[0]])
                ytrue = y_batch_cpu.float().data.numpy()[:,0]
                ypred = ypred_batch.cpu().float().data.numpy()[:,0]
                y_true_cnn[batch_ind*batch_size:batch_ind*batch_size+bsize] = ytrue
                y_pred_cnn[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred

    if test == 'hybrid' or 'gcn0':
        # GCN-0
        gcn0_dataset = GCN_Dataset(gcn0_test_path)
        # initialize testing data loader
        batch_count = len(gcn0_dataset) // batch_size
        gcn0_dataloader = DataListLoader(gcn0_dataset, batch_size=7, shuffle=False)
        # define model
        gcn0_model = GeometricDataParallel(GCN(in_channels=20, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
        # load checkpoint file
        gcn0_checkpoint = torch.load(gcn0_checkpoint_path, map_location=device)
        # model state dict
        gcn0_model_state_dict = gcn0_checkpoint.pop("model_state_dict")
        gcn0_model.load_state_dict(gcn0_model_state_dict, strict=False)
        test_data_hdf = h5py.File(gcn0_test_path, 'r')
        gcn0_model.eval()
        y_true_gcn0, y_pred_gcn0, pdbid_array = [], [], []
        with torch.no_grad():
            for batch in tqdm(gcn0_dataloader):
                data_list = []
                for dataset in batch:
                    pdbid = dataset[0]
                    pdbid_array.append(pdbid)
                    affinity = test_data_hdf[pdbid].attrs["affinity"].reshape(1,-1)
                    vdw_radii = (test_data_hdf[pdbid].attrs["van_der_waals"].reshape(-1, 1))
                    node_feats = np.concatenate([vdw_radii, test_data_hdf[pdbid][:, 3:22]], axis=1)
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) 
                    x = torch.from_numpy(node_feats).float()
                    y = torch.FloatTensor(affinity).view(-1, 1)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                    data_list.append(data)
                batch_data = [x for x in data_list if x is not None]
                y_ = gcn0_model(batch_data)
                y = torch.cat([x.y for x in data_list])
                y_true_gcn0.append(y.cpu().data.numpy())
                y_pred_gcn0.append(y_.cpu().data.numpy())
        y_true_gcn0 = np.concatenate(y_true_gcn0).reshape(-1, 1).squeeze(1)
        y_pred_gcn0 = np.concatenate(y_pred_gcn0).reshape(-1, 1).squeeze(1)

    if test == 'hybrid' or 'gcn1':
        # GCN-1
        gcn1_dataset = GCN_Dataset(gcn1_test_path)
        # initialize testing data loader
        batch_count = len(gcn0_dataset) // batch_size
        gcn1_dataloader = DataListLoader(gcn1_dataset, batch_size=7, shuffle=False)
        # define model
        gcn1_model = GeometricDataParallel(GCN(in_channels=20, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
        # load checkpoint file
        gcn1_checkpoint = torch.load(gcn1_checkpoint_path, map_location=device)
        # model state dict
        gcn1_model_state_dict = gcn1_checkpoint.pop("model_state_dict")
        gcn1_model.load_state_dict(gcn1_model_state_dict, strict=False)
        test_data_hdf = h5py.File(gcn1_test_path, 'r')
        gcn1_model.eval()
        y_true_gcn1, y_pred_gcn1, pdbid_array = [], [], []
        with torch.no_grad():
            for batch in tqdm(gcn1_dataloader):
                data_list = []
                for dataset in batch:
                    pdbid = dataset[0]
                    pdbid_array.append(pdbid)
                    affinity = test_data_hdf[pdbid].attrs["affinity"].reshape(1,-1)
                    vdw_radii = (test_data_hdf[pdbid].attrs["van_der_waals"].reshape(-1, 1))
                    node_feats = np.concatenate([vdw_radii, test_data_hdf[pdbid][:, 3:22]], axis=1)
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) 
                    x = torch.from_numpy(node_feats).float()
                    y = torch.FloatTensor(affinity).view(-1, 1)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                    data_list.append(data)
                batch_data = [x for x in data_list if x is not None]
                y_ = gcn1_model(batch_data)
                y = torch.cat([x.y for x in data_list])
                y_true_gcn1.append(y.cpu().data.numpy())
                y_pred_gcn1.append(y_.cpu().data.numpy())
        y_true_gcn1 = np.concatenate(y_true_gcn1).reshape(-1, 1).squeeze(1)
        y_pred_gcn1 = np.concatenate(y_pred_gcn1).reshape(-1, 1).squeeze(1)

    # compute metrics
    if test == 'hybrid':
        y_true = y_true_cnn/3 + y_true_gcn0/3 + y_true_gcn1/3
        y_pred = y_pred_cnn/3 + y_pred_gcn0/3 + y_pred_gcn1/3
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    if test == 'cnn':
        y_true = y_true_cnn
        y_pred = y_pred_cnn
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    if test == 'gcn0':
        y_true = y_true_gcn0
        y_pred = y_pred_gcn0
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    if test == 'gcn1':
        y_true = y_true_gcn1
        y_pred = y_pred_gcn1
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    # print test summary
    print('Test Summary:')
    print('RMSE: %.3f, MAE: %.3f, r^2 score: %.3f, Pearson r: %.3f, Spearman p: %.3f, Mean: %.3f, SD: %.3f' % (rmse, mae, r2, pearson, spearman, mean, std))

    #print scatterplot of true and predicted values
    plt.rcParams["figure.figsize"] = (7,7)
    plt.scatter(y_true, y_pred, color='darkslateblue', edgecolors='black')       # color = firebrick if VANILLA
    plt.xlabel("True", fontsize=20, weight='bold')
    plt.ylabel("Predicted", fontsize=20, weight='bold')
    plt.xticks([0,3,6,9,12], fontsize=18)
    plt.yticks([0,3,6,9,12], fontsize=18)
    plt.xlim(0,14)
    plt.ylim(0,14)
    plt.plot(np.arange(0,15), np.arange(0,15), color='dimgray', linestyle='-',zorder=0, linewidth=2)

    
""" Define a class to contain the data that will be included in the dataloader 
sent to the GCN model """

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


""" Define GCN class """

class GCN(torch.nn.Module):

    def __init__(self, in_channels, gather_width=128, prop_iter=4, dist_cutoff=3.5):
        super(GCN, self).__init__()

        #define distance cutoff
        self.dist_cutoff=torch.Tensor([dist_cutoff])
        if torch.cuda.is_available():
            self.dist_cutoff = self.dist_cutoff.cuda()

        #Attentional aggregation
        self.gate_net = nn.Sequential(nn.Linear(in_channels, int(in_channels/2)), nn.Softsign(), nn.Linear(int(in_channels/2), int(in_channels/4)), nn.Softsign(), nn.Linear(int(in_channels/4),1))
        self.attn_aggr = AttentionalAggregation(self.gate_net)
        
        #Gated Graph Neural Network
        self.gate = GatedGraphConv(in_channels, prop_iter, aggregation=self.attn_aggr)

        #Simple neural networks for use in asymmetric attentional aggregation
        self.attn_net_i=nn.Sequential(nn.Linear(in_channels * 2, in_channels), nn.Softsign(),nn.Linear(in_channels, gather_width), nn.Softsign())
        self.attn_net_j=nn.Sequential(nn.Linear(in_channels, gather_width), nn.Softsign())

        #Final set of linear layers for making affinity prediction
        self.output = nn.Sequential(nn.Linear(gather_width, int(gather_width / 1.5)), nn.ReLU(), nn.Linear(int(gather_width / 1.5), int(gather_width / 2)), nn.ReLU(), nn.Linear(int(gather_width / 2), 1))

    def forward(self, data):

        #Move data to GPU
        if torch.cuda.is_available():
            data.x = data.x.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.edge_index = data.edge_index.cuda()
            data.batch = data.batch.cuda()

        # allow nodes to propagate messages to themselves
        data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr.view(-1))

        # restrict edges to the distance cutoff
        row, col = data.edge_index
        mask = data.edge_attr <= self.dist_cutoff
        mask = mask.squeeze()
        row, col, edge_feat = row[mask], col[mask], data.edge_attr[mask]
        edge_index=torch.stack([row,col],dim=0)

        # propagation
        node_feat_0 = data.x
        node_feat_1 = self.gate(node_feat_0, edge_index, edge_feat)
        node_feat_attn = torch.nn.Softmax(dim=1)(self.attn_net_i(torch.cat([node_feat_1, node_feat_0], dim=1))) * self.attn_net_j(node_feat_0)

        # globally sum features and apply linear layers
        pool_x = global_add_pool(node_feat_attn, data.batch)
        prediction = self.output(pool_x)

        return prediction

""" Define a class to contain the extracted 3D-CNN features that will be included in the dataloader 
sent to the fully-connected network """

class MLP_Dataset(Dataset):
	def __init__(self, npy_path, feat_dim=22):
		super(MLP_Dataset, self).__init__()
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


""" Define fully-connected network class """
class MLP(nn.Module):
	def __init__(self, use_cuda=True):
		super(MLP, self).__init__()     
		self.use_cuda = use_cuda

		self.fc1 = nn.Linear(2048, 100)
		torch.nn.init.normal_(self.fc1.weight, 0, 1)
		self.fc1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.3).train()
		self.fc2 = nn.Linear(100, 1)
		torch.nn.init.normal_(self.fc2.weight, 0, 1)
		self.relu = nn.ReLU()

	def forward(self, x):
		fc1_z = self.fc1(x)
		fc1_y = self.relu(fc1_z)
		fc1 = self.fc1_bn(fc1_y) if fc1_y.shape[0]>1 else fc1_y  #batchnorm train require more than 1 batch
		fc2_z = self.fc2(fc1)
		return fc2_z, fc1_z
