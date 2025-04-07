import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import time

### Define a neural network model
#Structure:
#Input layer: 2 nodes corresponding to state-action pair $(s,a) \in \mathcal{S}\times\mathcal{A}$
#Output layer: 1 node corresponding to the vector value $\Lambda(s,a)$
#We initialize two neural networks, an approximator DNN: $\hat{\Lambda}(s,a,w)$ and a target DNN: $\Lambda^t(s,a,w)$

# define agent neural network class
class dnn(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, layers):
        super(dnn, self).__init__()

        # create input, hidden and output layers with ReLU activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_inputs, layers[0]))
        for i, l in enumerate(layers):
            self.layers.append(nn.ReLU())
            if i < len(layers)-1:
                self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.layers.append(nn.Linear(layers[-1], n_outputs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

# define the training configuration
def config_nn(dnn:dnn, optimizer_name:str, scheduler_name:str, lr_init:float, sgd_momentum=0.99):
    loss_fn = nn.MSELoss() # mean square loss
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(dnn.parameters(), lr=lr_init)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(dnn.parameters(), lr=lr_init)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(dnn.parameters(), lr=lr_init, momentum=sgd_momentum)

    if scheduler_name == 'LinearLR':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=0.005, total_iters=500)
    elif scheduler_name == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.99, patience=50, threshold=0.0001, threshold_mode='rel', 
            cooldown=0, min_lr=1e-5, eps=1e-08, verbose='deprecated')
    
    return loss_fn, optimizer, scheduler


def train_nn1(dnn:dnn, X:torch.tensor, Y:torch.tensor, loss_fn, optimizer, scheduler, 
                loss_coeffs, epochs, batch_size, shuffledata=False, allowPrint=False):
    
    n_batches = int((len(X)-1)/batch_size)+1
    training_loss = np.zeros(epochs)
    learning_rate = []
    coeff = loss_coeffs

    if allowPrint == True:
        print('\n---------------------------------')
        print(f'training data size = {len(X)}')
        print(f'training batch size = {batch_size}')
        print(f'n_epochs = {epochs}')
        print(f'n_batches = {n_batches}')
        print(f'loss weights = {coeff}')
        print('---------------------------------\n')
    
    # train on multiple epochs
    for ep in range(epochs):
        j = 0
        epoch_trainLoss1 = 0

        tStartEp = time.time()
        # update neural net parameters by batchwise training per epoch
        for i in range(n_batches):
            # set training mode
            dnn.train(True) #(optional)
            # compute necessary data
            if j+batch_size >= len(X):
                j_batch_end = len(X)
            else:
                j_batch_end = j+batch_size

            # predictions and targets
            x_train = X[j:j+j_batch_end,:]
            y_target = Y[j:j+j_batch_end,:]
            y_pred = dnn(x_train)

            # loss computations
            loss = loss_fn(y_pred, y_target)
            accuracy = torch.abs((y_pred.detach()-y_target))/torch.abs(y_target)
            # l1_reg = torch.tensor(0.)
            # for param in agent_nn.parameters():
            #     l1_reg += torch.sum(torch.abs(param))
            # l1_loss = l1_lambda*l1_reg
            # final loss
            loss = coeff[0]*loss
            # backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update loss and indices
            epoch_trainLoss1 += loss.detach().item()

            if ep == 0 and allowPrint==True:
                if i == 0:
                    print('batch_split')
                print(j, j_batch_end-1)
                if i == n_batches-1:
                    print('\n')
            j = j_batch_end
        
        tEnd_ep = time.time()
        tEp = tEnd_ep - tStartEp

        # update epoch training loss
        training_loss[ep] = epoch_trainLoss1 #/n_batches

        # Update learning rate every epoch
        current_lr = optimizer.param_groups[0]['lr']
        learning_rate.append(current_lr)
        scheduler.step(np.sum(training_loss[ep]))

        # print loss every few epochs
        if ep == 0:
            loss0 = training_loss[ep]
            # l1Loss0 = training_loss[ep,3]
            if allowPrint == True:
                print('\n---------------------------------')
                print(f'epoch: {ep}/{epochs}, \t loss: {loss0:.3f} \t rel:{1000:.1f}, \t time: {tEp:.2f}')
        if (ep+10)%1 == 0:
            loss = training_loss[ep]
            relLoss = loss/loss0*1000
            if allowPrint == True:
                print(f'epoch: {ep+1}/{epochs}, \t loss: {loss:.3f} \t rel:{relLoss:.1f}, \t time: {tEp:.2f}')
            if ep<epochs-1 and relLoss <= 1e-12:
                if allowPrint==True:
                    print(f'---------------------------------')
                break
            elif ep == epochs-1:
                if allowPrint == True:
                    print(f'Maximum Iterations Reached\n---------------------------------')
        stopping_epoch = ep

    return training_loss, learning_rate, stopping_epoch
