import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop, lr_scheduler
import numpy as np
from scipy import stats
from sklearn.metrics import *
import matplotlib.pyplot as plt

def train_3dcnn(train_hdf, val_hdf, checkpoint_dir, best_checkpoint_dir, previous_checkpoint = None, best_previous_checkpoint=None):
    '''
    Define a function to train the 3D-CNN model
    Inputs:
    1) train_hdf: training hdf file name
    2) val_hdf: validation hdf file name
    3) checkpoint_dir: path to save checkpoint file: 'path/to/file.pt'
    4) best_checkpoint_dir: path to save best checkpoint file: 'path/to/file.pt'
    5) previous_checkpoint: path to the checkpoint at which training should be started; default = None, i.e. training from scratch
    6) best_previous_checkpoint: path to the best checkpoint from the previous round of training (required); default = None, i.e. training from scratch
    Output:
    1) checkpoint file from the endpoint of the training
    2) checkpoint file from the epoch with highest average correlation on validation

    '''

    # define parameters
    batch_size = 50
    learning_rate = .0007
    learning_decay_iter=150
    epoch_count = 100

    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    # initialize Datasets
    dataset = CNN_Dataset(train_hdf)
    val_dataset = CNN_Dataset(val_hdf)

    # initialize Dataloaders
    batch_count = len(dataset.data_info_list) // batch_size + 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # define model and helper functions
    model = Model_3DCNN(use_cuda=use_cuda)
    model.to(device)
    loss_func = nn.MSELoss().float()
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.95)
    
    # initialize training variables
    epoch_start = 0
    step = 0
    epoch_train_losses, epoch_val_losses, epoch_avg_corr = [], [], []
    best_average_corr = float('-inf')

    #load previous checkpoint if applicable
    if previous_checkpoint!=None:
        best_checkpoint = torch.load(best_previous_checkpoint, map_location = device)
        torch.save(best_checkpoint, best_checkpoint_dir)
        best_average_corr = best_checkpoint["best_avg_corr"]
        checkpoint = torch.load(previous_checkpoint, map_location=device)
        model_state_dict = checkpoint.pop('model_state_dict')
        model.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        step=checkpoint['step']
        epoch_train_losses = checkpoint['epoch_train_losses']
        epoch_val_losses = checkpoint['epoch_val_losses']
        epoch_avg_corr = checkpoint['epoch_avg_corr']
        print('checkpoint loaded: %s' % previous_checkpoint)

    def validate_model():
        y_true_array = np.zeros((len(val_dataset),), dtype=np.float32)
        y_pred_array = np.zeros((len(val_dataset),), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for batch_ind, batch in enumerate(val_dataloader):
               
                # transfer to GPU
                x_batch_cpu, y_batch_cpu, _ = batch
                x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
                ypred_batch, _ = model(x_batch[:x_batch.shape[0]])
                
                # compute and print batch loss
                loss = loss_func(ypred_batch.cpu().float(), y_batch_cpu.float())
                print('[%d/%d-%d/%d] validation loss: %.3f' % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item()))
                
                #assemble the full datasets
                bsize = x_batch.shape[0]
                ytrue = y_batch_cpu.float().data.numpy()[:,0]
                ypred = ypred_batch.cpu().float().data.numpy()[:,0]
                y_true_array[batch_ind*batch_size:batch_ind*batch_size+bsize] = ytrue
                y_pred_array[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred
                
            #compute average correlation
            pearsonr = stats.pearsonr(y_true_array, y_pred_array)[0]
            spearmanr = stats.spearmanr(y_true_array, y_pred_array)[0]
            avg_corr = (pearsonr + spearmanr)/2
        
        # print information during training
        print('[%d/%d] validation results-- pearsonr: %.3f, spearmanr: %.3f, rmse: %.3f, mae: %.3f, r2: %.3f' % (epoch_ind+1, epoch_count, pearsonr, spearmanr, mean_squared_error(y_true_array, 
                                                                                          y_pred_array)**(1/2), mean_absolute_error(y_true_array, y_pred_array), r2_score(y_true_array, y_pred_array)))
        return mean_squared_error(y_true_array, y_pred_array), avg_corr

    for epoch_ind in range(epoch_start, epoch_count):
        x_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
        y_true_epoch = np.zeros((len(dataset),), dtype=np.float32)
        y_pred_epoch = np.zeros((len(dataset),), dtype=np.float32)
        for batch_ind, batch in enumerate(dataloader):
            model.train()

        # transfer to GPU and save in the epoch array
            x_batch_cpu, y_batch_cpu, _ = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
            bsize = x_batch.shape[0]
            ypred_batch, _ = model(x_batch[:x_batch.shape[0]])
            y_true_epoch[batch_ind*batch_size:batch_ind*batch_size+bsize] = y_batch_cpu.float().data.numpy()[:,0]
            y_pred_epoch[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred_batch.cpu().float().data.numpy()[:,0]

        # compute loss 
            loss = loss_func(ypred_batch.cpu().float(), y_batch_cpu.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            print("[%d/%d-%d/%d] training loss: %.3f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss))

        epoch_train_losses.append(mean_squared_error(y_true_epoch, y_pred_epoch))
        val_loss, average_corr = validate_model()
        epoch_val_losses.append(val_loss)
        epoch_avg_corr.append(average_corr)
        
        checkpoint_dict = {'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,'step': step,'epoch': epoch_ind,
                           'epoch_val_losses': epoch_val_losses,'epoch_train_losses': epoch_train_losses,'epoch_avg_corr' : epoch_avg_corr,'best_avg_corr': best_average_corr}

        if (average_corr > best_average_corr):
            best_average_corr = average_corr
            checkpoint_dict["best_avg_corr"] = best_average_corr
            torch.save(checkpoint_dict, best_checkpoint_dir)
            print("best checkpoint saved: %s" % best_checkpoint_dir)
        torch.save(checkpoint_dict, checkpoint_dir)
        print('checkpoint saved: %s' % checkpoint_dir)
    
    # create learning curve and correlation plot
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(1, epoch_count+1), np.array(epoch_train_losses), label = 'training')
    axs[0].plot(np.arange(1, epoch_count+1), np.array(epoch_val_losses), label = 'validation')
    axs[0].set_xlabel('Epoch', fontsize=20)
    axs[0].set_ylabel('Loss', fontsize=20)
    axs[0].legend(fontsize=18)
    axs[1].plot(np.arange(1, epoch_count+1), np.array(epoch_avg_corr))
    axs[1].set_xlabel('Epoch', fontsize=20)
    axs[1].set_ylabel('Validation Correlation', fontsize=20)
    axs[1].set_ylim(0,1)
    plt.show()

    # close dataset
    dataset.close()
    val_dataset.close()
