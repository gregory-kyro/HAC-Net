import numpy as np
from scipy.stats import *
from sklearn.metrics import *
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader

def savefeat_3dcnn(hdf_path, feature_length, checkpoint_path, npy_path):

    """
    Define a function to test the 3D CNN model and save relevant features
    Inputs:
    1) hdf_path: path/to/file.hdf
    2) feature length: length of the flattened output features
    3) checkpoint_path: path/to/checkpoint/file.pt
    4) npy_path: path/to/save/features.npy

    Output:
    1) numpy file containing the saved features, with the last column being the true affinity value.
    """

    # define parameters
    multi_gpus = False
    batch_size = 50
    device_name = "cuda:0"
    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    if use_cuda:
        device = torch.device(device_name)
        torch.cuda.set_device(int(device_name.split(':')[1]))
    else:   
        device = torch.device("cpu")
    print(use_cuda, cuda_count, device)
    # load testing 
     = CNN_(hdf_path)
    # check multi-gpus
    num_workers = 0
    if multi_gpus and cuda_count > 1:
        num_workers = cuda_count
    # initialize testing data loader
    batch_count = len() // batch_size
    dataloader = DataLoader(, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)
    # define model
    model = Model_3DCNN(use_cuda=use_cuda)
    if multi_gpus and cuda_count > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module
    # load checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # model state dict
    model_state_dict = checkpoint.pop("model_state_dict")
    model.load_state_dict(model_state_dict, strict=False)
    # create empty arrays to hold predicted and true values
    ytrue_arr = np.zeros((len(),), dtype=np.float32)
    ypred_arr = np.zeros((len(),), dtype=np.float32)
    flatfeat_arr = np.zeros((len(), feature_length + 1))
    pdbid_arr = np.zeros((len(),), dtype=object)
    pred_list = []
    model.eval()
    with torch.no_grad():
        for batch_ind, batch in enumerate(dataloader):
            # transfer to GPU
            x_batch_cpu, y_batch_cpu, pdbid_batch = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
            # arrange and filter
            bsize = x_batch.shape[0]
            ypred_batch, flatfeat_batch = model(x_batch[:x_batch.shape[0]])
            ytrue = y_batch_cpu.float().data.numpy()[:,0]
            ypred = ypred_batch.cpu().float().data.numpy()[:,0]
            flatfeat = flatfeat_batch.cpu().data.numpy()
            ytrue_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = ytrue
            ypred_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred
            flatfeat_arr[batch_ind*batch_size:batch_ind*batch_size+bsize, :-1] = flatfeat
            pdbid_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = pdbid_batch
    flatfeat_arr[:,-1] = ytrue_arr
    np.save(npy_path, flatfeat_arr)
