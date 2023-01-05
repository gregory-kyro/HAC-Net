import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy
from scipy import stats
from sklearn.metrics import *
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import RMSprop, lr_scheduler
from torch.utils.data import DataLoader

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


"""Define a function to train the fully-connected network with extracted features from 3D-CNN"""

def train_MLP(input_train_data, input_val_data, checkpoint_dir, best_checkpoint_dir, learning_decay_iter = 150, load_previous_checkpoint = False, previous_checkpoint = None, best_previous_checkpoint = None):
    """
    Inputs:
    1) input_train_data: path to train.npy data
    2) input_val_data: path to val.npy data
    3) checkpoint_dir: path to save checkpoint file: 'path/to/file.pt'
    4) best_checkpoint_dir: path to save best checkpoint file: 'path/to/file.pt'
    5) learning_decay_iter: frequency at which the learning rate is decreased by a multiplicative factor of decay_rate; set to 150 by default
    6) load_previous_checkpoint: boolean variable indicating whether or not training should be started from an existing checkpoint. False by default
    7) previous_checkpoint: path to the checkpoint at which training should be started. None by default.
    8) best_previous_checkpoint: path to the checkpoint which was saved as "best" in the previous training. None by default
    Outputs:
    1) checkpoint file from the endpoint of the training
    2) best checkpoint file, defined as that which maximizes the average of pearson and spearman correlations obtained from the validation data
    """

    # define parameters
    batch_size = 50
    learning_rate = .0007
    decay_iter = learning_decay_iter 
    decay_rate = 0.95
    epoch_count = 100
    checkpoint_iter = 343
    device_name = "cuda:0"
    multi_gpus = False

    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    if use_cuda:
        device = torch.device(device_name)
        torch.cuda.set_device(int(device_name.split(':')[1]))
    else:
        device = torch.device("cpu")
    print(use_cuda, cuda_count, device)

    def worker_init_fn(worker_id):
        np.random.seed(int(0))

    # build training dataset variable
    dataset = MLP_Dataset(input_train_data)

    # build validation dataset variable
    val_dataset = MLP_Dataset(input_val_data)

    # check multi-gpus
    num_workers = 0
    if multi_gpus and cuda_count > 1:
        num_workers = cuda_count

    def validate_model():
        loss_fn = nn.MSELoss().float()
        ytrue_arr = np.zeros((len(val_dataset),), dtype=np.float32)
        ypred_arr = np.zeros((len(val_dataset),), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for batch_ind, batch in enumerate(val_dataloader):
        # transfer to GPU
                x_batch_cpu, y_batch_cpu = batch
                x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
                ypred_batch, _ = model(x_batch[:x_batch.shape[0]])
        # compute and print batch loss
                loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
        #assemble the full datasets
                bsize = x_batch.shape[0]
                ytrue = y_batch_cpu.float().data.numpy()[:,0]
                ypred = ypred_batch.cpu().float().data.numpy()[:,0]
                ytrue_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = ytrue
                ypred_arr[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred            
        #compute average correlation
            pearsonr = stats.pearsonr(ytrue_arr, ypred_arr)[0]
            spearmanr = stats.spearmanr(ytrue_arr, ypred_arr)[0]
            avg_corr = (pearsonr + spearmanr)/2
        return mean_squared_error(ytrue_arr, ypred_arr), avg_corr

    # initialize training data loader
    batch_count = len(dataset) // batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=None, shuffle=True)

    # initialize validation data loader
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

    # define model
    model = MLP(use_cuda=use_cuda)
    if multi_gpus and cuda_count > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module

    # define loss
    loss_fn = nn.MSELoss().float()
    # define optimizer
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    # define scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_iter, gamma=decay_rate)
    # train model
    epoch_start = 0
    step = 0
    epoch_train_losses, epoch_val_losses, epoch_avg_corr = [], [], []
    best_average_corr = float('-inf')
    best_checkpoint_dict = None
    #load previous checkpoint if applicable
    if load_previous_checkpoint:
        best_checkpoint = torch.load(best_previous_checkpoint, map_location = device)
        best_checkpoint_dict = best_checkpoint.pop("model_state_dict")
        best_average_corr = best_checkpoint["best_avg_corr"]
        checkpoint = torch.load(previous_checkpoint, map_location=device)
        model_state_dict = checkpoint.pop("model_state_dict")
        model.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        loss = checkpoint["loss"]
        epoch_train_losses = checkpoint["epoch_train_losses"]
        epoch_val_losses = checkpoint["epoch_val_losses"]
        epoch_avg_corr = checkpoint["epoch_avg_corr"]
    for epoch_ind in range(epoch_start, epoch_count):
        losses = []
        for batch_ind, batch in enumerate(dataloader):
            model.train()
        # transfer to GPU
            x_batch_cpu, y_batch_cpu = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
            ypred_batch, _ = model(x_batch[:x_batch.shape[0]])
        # compute loss
            loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
            losses.append(loss.cpu().data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            val_loss, average_corr = validate_model()
            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "step": step,
                "epoch": epoch_ind,
                "epoch_val_losses": epoch_val_losses,
                "epoch_train_losses": epoch_train_losses,
                "epoch_avg_corr" : epoch_avg_corr,
                "best_avg_corr" : best_average_corr 
            }
            if (average_corr > best_average_corr):
                best_average_corr = average_corr
                checkpoint_dict["best_avg_corr"] = best_average_corr
                best_checkpoint_dict = checkpoint_dict
                torch.save(best_checkpoint_dict, best_checkpoint_dir)
            torch.save(checkpoint_dict, checkpoint_dir)
        step += 1
	print('Epoch: ', step)
    val_loss, average_corr = validate_model()
    epoch_train_losses.append(np.mean(losses))
    epoch_val_losses.append(val_loss)
    epoch_avg_corr.append(average_corr)
    checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "step": step,
                "epoch": epoch_ind,
                "epoch_val_losses": epoch_val_losses,
                "epoch_train_losses": epoch_train_losses,
                "epoch_avg_corr" : epoch_avg_corr,
                "best_avg_corr": best_average_corr
            }
    if (average_corr > best_average_corr):
        best_average_corr = average_corr
        checkpoint_dict["best_avg_corr"] = best_average_corr
        best_checkpoint_dict = checkpoint_dict
        torch.save(best_checkpoint_dict, best_checkpoint_dir)
    torch.save(checkpoint_dict, checkpoint_dir)
