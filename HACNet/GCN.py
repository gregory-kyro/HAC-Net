import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, dense_to_sparse
from torch_geometric.nn.aggr import AttentionalAggregation
from torch._C import NoneType
from torch.optim import Adam
from torch_geometric.data import Data, DataListLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch.utils.data import Dataset
import numpy as np
import h5py
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, pairwise_distances
import os
import random
from scipy import stats
import matplotlib.pyplot as plt

''' Define a class to contain the data that will be included in the dataloader 
sent to the GCN model '''

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


''' Define GCN class '''

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


''' Define train function for MP-GCN'''
def train_gcn(train_data, val_data, checkpoint_name, best_checkpoint_name, load_checkpoint_path = None, best_previous_checkpoint=None):

    '''
    Inputs:
    1) train_data: training hdf file name
    2) val_data: validation hdf file name
    3) checkpoint_name: path to save checkpoint_name.pt
    4) best_checkpoint_name: path to save best_checkpoint_name.pt
    5) load_checkpoint_path: path to checkpoint file to load; default is None, i.e. training from scratch
    6) best_previous_checkpoint: path to the best checkpoint from the previous round of training (required); default is None, i.e. training from scratch
    Output:
    1) checkpoint file, to load into testing function; saved as: checkpoint_name
    '''

    # define train and validation hdf files
    train_data_hdf = h5py.File(train_data, 'r')
    val_data_hdf = h5py.File(val_data, 'r')

    # define parameters
    epochs = 300                   # number of training epochs
    batch_size = 7                 # batch size to use for training
    learning_rate = 0.001          # learning rate to use for training
    gather_width = 128             # gather width
    prop_iter = 4                  # number of propagation interations
    dist_cutoff = 3.5              # common cutoff for donor-to-acceptor distance for energetically significant H bonds in proteins is 3.5 Å
    feature_size = 20              # number of features: 19 + Van der Waals radius

    # seed all random number generators and set cudnn settings for deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = '0'

    def worker_init_fn(worker_id):
        np.random.seed(int(0))
        
    # initialize checkpoint parameters
    checkpoint_epoch = 0
    checkpoint_step = 0
    epoch_train_losses, epoch_val_losses, epoch_avg_corr = [], [], []
    best_average_corr = float('-inf')

    # define function to return checkpoint dictionary
    def checkpoint_model(model, dataloader, epoch, step):
        validate_dict = validate(model, dataloader)
        model.train()
        checkpoint_dict = {'model_state_dict': model.state_dict(), 'step': step, 'epoch': epoch, 'validate_dict': validate_dict,
                           'epoch_train_losses': epoch_train_losses, 'epoch_val_losses': epoch_val_losses, 'epoch_avg_corr': epoch_avg_corr, 'best_avg_corr': best_average_corr}
        torch.save(checkpoint_dict, checkpoint_name)
        return checkpoint_dict

    # define function to perform validation
    def validate(model, val_dataloader):
        # initialize
        model.eval()
        y_true = np.zeros((len(val_dataset),), dtype=np.float32)
        y_pred = np.zeros((len(val_dataset),), dtype=np.float32)
        # validation
        for batch_ind, batch in enumerate(val_dataloader):
            data_list = []
            for dataset in batch:
                pdbid = dataset[0]
                affinity = val_data_hdf[pdbid].attrs['affinity'].reshape(1,-1)
                vdw_radii = (val_data_hdf[pdbid].attrs['van_der_waals'].reshape(-1, 1))
                node_feats = np.concatenate([vdw_radii, val_data_hdf[pdbid][:, 3:22]], axis=1)
                edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float())
                x = torch.from_numpy(node_feats).float()
                y = torch.FloatTensor(affinity).view(-1, 1)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                data_list.append(data)
            batch_data = [x for x in data_list]
            y_ = model(batch_data)
            y = torch.cat([x.y for x in data_list])
            y_true[batch_ind*batch_size:batch_ind*batch_size+7] = y.cpu().float().data.numpy()[:,0]
            y_pred[batch_ind*batch_size:batch_ind*batch_size+7] = y_.cpu().float().data.numpy()[:,0]
            loss = criterion(y.float(), y_.cpu().float())
            print('[%d/%d-%d/%d] validation loss: %.3f' % (epoch+1, epochs, batch_ind+1, len(val_dataset)//batch_size, loss))

        # compute r^2
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        # compute mae
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        # compute mse
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        # compute pearson correlation coefficient
        pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))[0]
        # compte spearman correlation coefficient
        spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))[0]
        # write out metrics
        print('r2: {}\tmae: {}\trmse: {}\tpearsonr: {}\t spearmanr: {}'.format(r2, mae, mse**(1/2), pearsonr, spearmanr))
        epoch_val_losses.append(mse)
        epoch_avg_corr.append((pearsonr+spearmanr)/2)
        model.train()
        return {'r2': r2, 'mse': mse, 'mae': mae, 'pearsonr': pearsonr, 'spearmanr': spearmanr,
                'y_true': y_true, 'y_pred': y_pred, 'best_average_corr': best_average_corr}
   
    # construct model
    model = GeometricDataParallel(GCN(in_channels=feature_size, gather_width=gather_width, prop_iter=prop_iter, dist_cutoff=dist_cutoff)).float()

    train_dataset = GCN_Dataset(data_file=train_data)
    val_dataset = GCN_Dataset(data_file=val_data)
        
    # construct training and validation dataloaders to be fed to model
    batch_count=len(train_dataset)
    train_dataloader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=True)
    val_dataloader = DataListLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=True)

    # load checkpoint file
    if load_checkpoint_path != None:
        if torch.cuda.is_available():
            model_train_dict = torch.load(load_checkpoint_path)
            best_checkpoint = torch.load(best_previous_checkpoint)
        else:
            model_train_dict = torch.load(load_checkpoint_path, map_location=torch.device('cpu'))
            best_checkpoint = torch.load(best_previous_checkpoint, map_location = torch.device('cpu'))
        model.load_state_dict(model_train_dict['model_state_dict'])
        checkpoint_epoch = model_train_dict['epoch']
        checkpoint_step = model_train_dict['step']
        epoch_train_losses = model_train_dict['epoch_train_losses']
        epoch_val_losses = model_train_dict['epoch_val_losses']
        epoch_avg_corr = model_train_dict['epoch_avg_corr']
        val_dict = model_train_dict['validate_dict']
        torch.save(best_checkpoint, best_checkpoint_name)
        best_average_corr = best_checkpoint["best_avg_corr"]
        
    model.train()
    model.to(0)
    
    # set loss as MSE
    criterion = nn.MSELoss().float()
    # set Adam optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate) 
    
    # train model
    step = checkpoint_step
    for epoch in range(checkpoint_epoch, epochs):
        y_true = np.zeros((len(train_dataset),), dtype=np.float32)
        y_pred = np.zeros((len(train_dataset),), dtype=np.float32)
        for batch_ind, batch in enumerate(train_dataloader):
            data_list = []
            pdbid_array = []
            for dataset in batch:
                pdbid = dataset[0]
                pdbid_array.append(pdbid)
                affinity = train_data_hdf[pdbid].attrs['affinity'].reshape(1,-1)
                vdw_radii = (train_data_hdf[pdbid].attrs['van_der_waals'].reshape(-1, 1))
                node_feats = np.concatenate([vdw_radii, train_data_hdf[pdbid][:, 3:22]], axis=1)
                edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) 
                x = torch.from_numpy(node_feats).float()
                y = torch.FloatTensor(affinity).view(-1, 1)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                data_list.append(data)

            optimizer.zero_grad()
            batch_data = [x for x in data_list]
            y_ = model(batch_data)
            y = torch.cat([x.y for x in data_list])
            y_true[batch_ind*batch_size:batch_ind*batch_size+7] = y.cpu().float().data.numpy()[:,0]
            y_pred[batch_ind*batch_size:batch_ind*batch_size+7] = y_.cpu().float().data.numpy()[:,0]

            # compute loss and update parameters
            loss = criterion(y.float(), y_.cpu().float())
            loss.backward()
            optimizer.step()
            step += 1
            print("[%d/%d-%d/%d] training loss: %.3f" % (epoch+1, epochs, batch_ind+1, len(train_dataset)//batch_size, loss))

        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse=mean_squared_error(y_true,y_pred)
        epoch_train_losses.append(mse)
        pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
        spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

        # write training summary for the epoch
        print('epoch: {}\trmse:{:0.4f}\tr2: {:0.4f}\t pearsonr: {:0.4f}\tspearmanr: {:0.4f}\tmae: {:0.4f}\tpred'.format(epoch+1, mse**(1/2), r2, float(pearsonr[0]),
                    float(spearmanr[0]), float(mae)))
        
        checkpoint_dict = checkpoint_model(model, val_dataloader, epoch+1, step)
        if (checkpoint_dict["validate_dict"]["pearsonr"] + checkpoint_dict["validate_dict"]["spearmanr"])/2 > best_average_corr:
          best_average_corr = (checkpoint_dict["validate_dict"]["pearsonr"] + checkpoint_dict["validate_dict"]["spearmanr"])/2
          torch.save(checkpoint_dict, best_checkpoint_name)
        torch.save(checkpoint_dict, checkpoint_name)
          
    # learning curve and correlation plot
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(1, epochs+1), np.array(epoch_train_losses), label = 'training')
    axs[0].plot(np.arange(1, epochs+1), np.array(epoch_val_losses), label = 'validation')
    axs[0].set_xlabel('Epoch', fontsize=20)
    axs[0].set_ylabel('Loss', fontsize=20)
    axs[0].legend(fontsize=18)
    axs[1].plot(np.arange(1, epochs+1), np.array(epoch_avg_corr))
    axs[1].set_xlabel('Epoch', fontsize=20)
    axs[1].set_ylabel('Validation Correlation', fontsize=20)
    axs[1].set_ylim(0,1)
    plt.show()
  
    train_data_hdf.close()
    val_data_hdf.close()
