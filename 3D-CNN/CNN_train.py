from torch.utils.data import DataLoader
from torch.optim import RMSprop, lr_scheduler
import numpy as np
from scipy import stats
from sklearn.metrics import *
import matplotlib.pyplot as plt

def train_3dcnn(data_dir, train_hdf, val_hdf, checkpoint_dir, learning_decay_iter = 150, load_previous_checkpoint = False, previous_checkpoint = None, save_intermittently=False):
    '''
    Define a function to train the 3D-CNN model

    Inputs:
    1) data_dir: path to hdf data
    2) train_hdf: training hdf file name
    3) val_hdf: validation hdf file name
    4) checkpoint_dir: path to save checkpoint file: 'path/to/file.pt'
    5) learning_decay_iter: frequency at which the learning rate is decreased by a multiplicative factor of 0.95; default = 150
    6) load_previous_checkpoint: boolean variable indicating whether or not training should be started from an existing checkpoint; default = False
    7) previous_checkpoint: path to the checkpoint at which training should be started; default = False
    8) save_intermittently: boolean as to whether to save checkpoints in the middle of epochs; default=False

    Output:
    1) checkpoint file from the endpoint of the training

    '''

    # define parameters
    batch_size = 50
    learning_rate = .0007
    epoch_count = 200
    checkpoint_iter = 343
    device_name = 'cuda:0'

    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    device = torch.device(device_name)
    torch.cuda.set_device(int(device_name.split(':')[1]))

    def worker_init_fn(worker_id):
        np.random.seed(int(0))

    # build training dataset variable
    dataset = CNN_Dataset(os.path.join(data_dir, train_hdf))

    # build validation dataset variable
    val_dataset = CNN_Dataset(os.path.join(data_dir, val_hdf))

    def validate_model():
        loss_func = nn.MSELoss().float()
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
                print('[%d/%d-%d/%d] validation, loss: %.3f' % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item()))
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
        print('overall validation corr: %.3f, average loss: %.3f' % (avg_corr, mean_squared_error(y_true_array, y_pred_array)))
        return mean_squared_error(y_true_array, y_pred_array), avg_corr

    # initialize training data loader
    batch_count = len(dataset.data_info_list) // batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, worker_init_fn=None, shuffle=True)
    # initialize validation data loader
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, worker_init_fn=None, shuffle=False)
    # define model
    model = Model_3DCNN(use_cuda=use_cuda)
    model.to(device)
    # define loss
    loss_func = nn.MSELoss().float()
    # define optimizer
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    # define scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.95)
    # train model
    epoch_start = 0
    step = 0
    epoch_train_losses, epoch_val_losses, epoch_avg_corr = [], [], []

    #load previous checkpoint if applicable
    if load_previous_checkpoint:
        checkpoint = torch.load(previous_checkpoint, map_location=device)
        model_state_dict = checkpoint.pop('model_state_dict')
        model.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        loss = checkpoint['loss']
        epoch_train_losses = checkpoint['epoch_train_losses']
        epoch_val_losses = checkpoint['epoch_val_losses']
        epoch_avg_corr = checkpoint['epoch_avg_corr']
        print('checkpoint loaded: %s' % previous_checkpoint)

    for epoch_ind in range(epoch_start, epoch_count):
        x_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
        losses = []
        for batch_ind, batch in enumerate(dataloader):
            model.train()
        # transfer to GPU
            x_batch_cpu, y_batch_cpu, _ = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
            ypred_batch, _ = model(x_batch[:x_batch.shape[0]])
        # compute loss
            loss = loss_func(ypred_batch.cpu().float(), y_batch_cpu.float())
            losses.append(loss.cpu().data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # print batch loss, then save last and best checkpoints if applicable
            print('[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f' % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))
            if save_intermittently and step % checkpoint_iter == 0:
                val_loss, average_corr = validate_model()
                checkpoint_dict = {'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,'step': step,'epoch': epoch_ind,
                                   'epoch_val_losses': epoch_val_losses,'epoch_train_losses': epoch_train_losses,'epoch_avg_corr' : epoch_avg_corr}
                torch.save(checkpoint_dict, checkpoint_dir)
                print('checkpoint saved: %s' % checkpoint_dir)
            step += 1

        val_loss, average_corr = validate_model()
        #Print epoch training and validation losses, then save last and best checkpoints
        print('[%d/%d] training, epoch loss: %.3f' % (epoch_ind+1, epoch_count, np.mean(losses)))
        epoch_train_losses.append(np.mean(losses))
        print('[%d/%d] validation, epoch loss: %.3f' % (epoch_ind+1, epoch_count, val_loss))
        epoch_val_losses.append(val_loss)
        epoch_avg_corr.append(average_corr)
        
        checkpoint_dict = {'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,'step': step,'epoch': epoch_ind,
                           'epoch_val_losses': epoch_val_losses,'epoch_train_losses': epoch_train_losses,'epoch_avg_corr' : epoch_avg_corr,'best_avg_corr': best_average_corr}

        torch.save(checkpoint_dict, checkpoint_dir)
        print('checkpoint saved: %s' % checkpoint_dir)
    
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

    # close dataset
    dataset.close()
    if (val_dataset):
        val_dataset.close()
