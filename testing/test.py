import h5py
import numpy as np
import math
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import DataListLoader
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp
from scipy.stats import *
from sklearn.metrics import *

def test_hybrid(test='hybrid',cnn_test_path, gcn0_test_path, gcn1_test_path, cnn_checkpoint_path, gcn0_checkpoint_path, gcn1_checkpoint_path):

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

    if test = 'hybrid' or 'cnn':
        # 3D-CNN
        batch_size = 50
        # load dataset
        cnn_dataset = Dataset_npy(cnn_test_path)
        # initialize data loader
        batch_count = len(cnn_dataset) // batch_size
        cnn_dataloader = DataLoader(cnn_dataset, batch_size=bacth_size, shuffle=False, num_workers=0, worker_init_fn=None)
        # define model
        cnn_model = Model_Linear(use_cuda=use_cuda)
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

    if test = 'hybrid' or 'gcn0':
        # MP-GCN-0
        gcn0_dataset = GCN_Dataset(gcn0_test_path, output_info = True)
        # initialize testing data loader
        batch_count = len(gcn0_dataset) // batch_size
        gcn0_dataloader = DataListLoader(gcn_dataset, batch_size=7, shuffle=False)
        # define model
        gcn0_model = GeometricDataParallel(MP_GCN(in_channels=20, out_channels=1, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
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

    if test = 'hybrid' or 'gcn1':
        # MP-GCN-1
        gcn1_dataset = GCN_Dataset(gcn1_test_path, output_info = True)
        # initialize testing data loader
        batch_count = len(gcn0_dataset) // batch_size
        gcn1_dataloader = DataListLoader(gcn1_dataset, batch_size=7, shuffle=False)
        # define model
        gcn1_model = GeometricDataParallel(MP_GCN(in_channels=20, out_channels=1, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
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
    if test = 'hybrid':
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

    if test = 'cnn':
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

    if test = 'gcn0':
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

    if test = 'gcn1':
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
    plt.scatter(y_true, y_pred, color='darkslateblue', edgecolors='black')
    plt.xlabel("True", fontsize=20, weight='bold')
    plt.ylabel("Predicted", fontsize=20, weight='bold')
    plt.xticks([0,3,6,9,12], fontsize=18)
    plt.yticks([0,3,6,9,12], fontsize=18)
    plt.xlim(0,14)
    plt.ylim(0,14)
    plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], color='dimgray', linestyle='-',zorder=0, linewidth=2)
