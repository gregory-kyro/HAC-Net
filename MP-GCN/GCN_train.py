import os
import numpy as np
import h5py
import random
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
from torch._C import NoneType
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, DataListLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt

''' Define train function for MP-GCN'''
def train_gcn(data_dir, train_data, val_data, checkpoint_dir, checkpoint_name, load_checkpoint_bool=False, load_checkpoint_path = None, checkpoint_mid_epoch=False):

    '''
    Inputs:
    1) data_dir: path to hdf files
    2) train_data: training hdf file name
    3) val_data: validation hdf file name
    4) checkpoint_dir: path/to/checkpoint/location/
    5) checkpoint_name: checkpoint_name.pt
    5) load_checkpoint_bool: boolean flag for whether to start training from a prior checkpoint; default is False
    6) load_checkpoint_path: path to checkpoint file to load; default is None
    7) checkpoint_mid_epoch: boolean flag for whether to save checkpoints in the middle of epochs, as opposed to only at epoch end; default is False

    Output:
    1) checkpoint file, to load into testing function; saved as: checkpoint_dir + checkpoint_name
    '''

    # set directory to path containing hdf files
    os.chdir(data_dir)

    # define train and validation hdf files
    train_data_hdf = h5py.File(train_data, 'r')
    val_data_hdf = h5py.File(val_data, 'r')

    # define parameters
    checkpoint = True              # boolean flag for checkpoints
    checkpoint_iter = 100          # number of batches per checkpoint, if checkpoint_mid_epoch=True
    num_workers = 24               # number of workers for datloader

    epochs = 220                   # number of training epochs
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

    # define function to return checkpoint dictionary
    def checkpoint_model(model, dataloader, epoch, step):
        validate_dict = validate(model, dataloader)
        model.train()
        checkpoint_dict = {'model_state_dict': model.state_dict(), 'args': NoneType, 'step': step, 'epoch': epoch, 'validate_dict': validate_dict,
                           'epoch_train_losses': epoch_train_losses, 'epoch_val_losses': epoch_val_losses, 'epoch_avg_corr': epoch_avg_corr}
        torch.save(checkpoint_dict, checkpoint_dir + checkpoint_name)
        # return the computed metrics so it can be used to update the training loop
        return checkpoint_dict

    # define function to perform validation
    def validate(model, val_dataloader):
        # initialize
        model.eval()
        y_true = []
        y_pred = []
        pdbid_list = []
        pose_list = []
        # validation
        for batch in tqdm(val_dataloader):
            data_list = []
            for dataset in batch:
                pdbid = dataset[0]
                affinity = val_data_hdf[pdbid].attrs['affinity'].reshape(1,-1)
                vdw_radii = (val_data_hdf[pdbid].attrs['van_der_waals'].reshape(-1, 1))
                node_feats = np.concatenate([vdw_radii, val_data_hdf[pdbid][:, 3:22]], axis=1)
                edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) ##--RAFI I don't see how this is helpful since pariwise_distances should return a positive value for everything except the main diagonal, so it's not sparse at all.
                x = torch.from_numpy(node_feats).float()
                y = torch.FloatTensor(affinity).view(-1, 1)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                data_list.append(data)
            if len(data_list) < 1:
                print('empty batch, skipping to next batch')
                continue

            batch_data = [x for x in data_list if x is not None]
            y_ = model(batch_data)
            y = torch.cat([x.y for x in data_list])
            pdbid_list.extend([x[0] for x in batch])
            y_true.append(y.cpu().data.numpy())
            y_pred.append(y_.cpu().data.numpy())

        y_true = np.concatenate(y_true).reshape(-1, 1)
        y_pred = np.concatenate(y_pred).reshape(-1, 1)

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
        tqdm.write(str(
                'r2: {}\tmae: {}\trmse: {}\tpearsonr: {}\t spearmanr: {}'.format(r2, mae, mse**(1/2), pearsonr, spearmanr)))
        epoch_val_losses.append(mse)
        epoch_avg_corr.append((pearsonr+spearmanr)/2)
        model.train()
        return {'r2': r2, 'mse': mse, 'mae': mae, 'pearsonr': pearsonr, 'spearmanr': spearmanr,
                'y_true': y_true, 'y_pred': y_pred, 'pdbid': pdbid_list}
   
    # construct model
    model = GeometricDataParallel(MP_GCN(in_channels=feature_size, out_channels=1, gather_width=gather_width, prop_iter=prop_iter, dist_cutoff=dist_cutoff)).float()

    train_dataset = GCN_Dataset(data_file=train_data, output_info=True)
    val_dataset = GCN_Dataset(data_file=val_data, output_info=True)
        
    # construct training and validation dataloaders to be fed to model
    train_dataloader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=True)
    val_dataloader = DataListLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=True)
    
    # print statement of complexes for sanity check
    tqdm.write('{} complexes in training dataset'.format(len(train_dataset)))
    tqdm.write('{} complexes in validation dataset'.format(len(val_dataset)))

    # initialize checkpoint parameters
    checkpoint_epoch = 0
    checkpoint_step = 0
    epoch_train_losses, epoch_val_losses, epoch_avg_corr = [], [], []

    # load checkpoint file
    if load_checkpoint_bool:
        if torch.cuda.is_available():
            model_train_dict = torch.load(load_checkpoint_path)
        else:
            model_train_dict = torch.load(load_checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_train_dict['model_state_dict'])
        checkpoint_epoch = model_train_dict['epoch']
        checkpoint_step = model_train_dict['step']
        epoch_train_losses = model_train_dict['epoch_train_losses']
        epoch_val_losses = model_train_dict['epoch_val_losses']
        epoch_avg_corr = model_train_dict['epoch_avg_corr']
        val_dict = model_train_dict['validate_dict']
        
    model.train()
    model.to(0)
    
    # set loss as MSE
    criterion = nn.MSELoss().float()
    # set Adam optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate) 
    
    # train model
    step = checkpoint_step
    for epoch in range(checkpoint_epoch, epochs):
        losses = []
        for batch in tqdm(train_dataloader):
            data_list = []
            pdbid_array = []
            for dataset in batch:
                pdbid = dataset[0]
                pdbid_array.append(pdbid)
                affinity = train_data_hdf[pdbid].attrs['affinity'].reshape(1,-1)
                vdw_radii = (train_data_hdf[pdbid].attrs['van_der_waals'].reshape(-1, 1))
                node_feats = np.concatenate([vdw_radii, train_data_hdf[pdbid][:, 3:22]], axis=1)
                edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) ##--RAFI I don't see how this is helpful since pariwise_distances should return a positive value for everything except the main diagonal, so it's not sparse at all.
                x = torch.from_numpy(node_feats).float()
                y = torch.FloatTensor(affinity).view(-1, 1)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                data_list.append(data)
            if len(data_list) < 1:
                print('empty batch, skipping to next batch')
                continue

            optimizer.zero_grad()
            batch_data = [x for x in data_list]
            y_ = model(batch_data)
            y = torch.cat([x.y for x in data_list])

            # compute loss
            loss = criterion(y.float(), y_.cpu().float())
            losses.append(loss.cpu().data.item())
            loss.backward()
            y_true = y.cpu().data.numpy()
            y_pred = y_.cpu().data.numpy()
            # compute r^2
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            # compute mae
            mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
            # compute pearson correlation coefficient
            pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
            # compute spearman correlation coefficient
            spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
            # write training summary for each epoch
            tqdm.write('epoch: {}\tloss:{:0.4f}\tr2: {:0.4f}\t pearsonr: {:0.4f}\tspearmanr: {:0.4f}\tmae: {:0.4f}\tpred stdev: {:0.4f}'
                        '\t pred mean: {:0.4f}'.format(epoch, loss.cpu().data.numpy(), r2, float(pearsonr[0]),
                        float(spearmanr[0]), float(mae), np.std(y_pred), np.mean(y_pred)))
            
            optimizer.step()
            step += 1
            if step%2500==0:
                output.clear()
               
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
