import numpy as np
import matplotlib.pyplot as plt
import LSE_net
import torch
import importlib
import pickle
import datetime

def myLog(x, base):
    return np.log(x)/np.log(base)

def createBetaArray(min, max, grow):
    beta = min
    beta_array = [beta]
    while beta <= max:
        beta = beta*grow
        beta_array.append(beta)
    beta_array = np.array(beta_array)
    return beta_array

def logSumExp(D_array, beta):
    D_min = np.min(D_array)
    D_off = D_array - D_min
    F = -1/beta * np.log(np.sum(np.exp(-beta*D_off))) + 1/beta * np.log(len(D_array)) + D_min
    # F = -np.log(np.sum(np.exp(-beta*D_off))) + np.log(len(D_array)) + D_min * beta
    return F

def gibbs(D_array, beta):
    D_min = np.min(D_array)
    D_off = D_array - D_min
    exp_bD = np.exp(-beta * D_off)
    sum_exp_bD = np.sum(exp_bD)
    P = exp_bD/sum_exp_bD
    return P

def area_approx_F(D_min, D_max_range, N_D, beta):
    min_beta_D_arr = beta * D_min
    x_max = beta * D_max_range - min_beta_D_arr
    F_est = -1/beta * np.log(N_D/x_max * (1 - np.exp(-x_max))) + D_min + 1/beta * np.log(N_D)
    # F_est = -1/beta * np.log(1/x_max * (1 - np.exp(-x_max))) + D_min
    return F_est

# function to generate logsum dataset
def logsum_dataset(n_curves, b_min, b_max, b_grow, D_min_range, D_max_scale, len_Darray_range, approx_log_sum_flr, io_scale):
    b_array = createBetaArray(b_min, b_max, b_grow)
    n_beta_per_curve = len(b_array)
    n_data = n_curves * n_beta_per_curve
    In = torch.zeros((n_data, 3), dtype=torch.float32)
    Out = torch.zeros((n_data,1), dtype=torch.float32)
    len_D_ARR = []

    cnt = 0
    for c in range(n_curves):
        
        len_Darray = int(np.random.uniform(len_Darray_range[0], len_Darray_range[1]))
        D0 = np.random.uniform(D_min_range[0], D_min_range[1])
        D_array = np.random.uniform(D0, D_max_scale, len_Darray)
        D_min = np.min(D_array)
        if approx_log_sum_flr:
            F_min = area_approx_F(D_min, D_max_scale, len_Darray, b_min)
        else:
            F_min = logSumExp(D_array, b_min)
        F_min = io_scale * F_min
        F_max = io_scale * D_min

        for b in b_array:
            Fb = io_scale * logSumExp(D_array, b)
            # In[cnt,:] = torch.tensor([F_min, F_max, b])
            In[cnt,:] = torch.tensor([F_min, F_max, myLog(b,10)])
            Out[cnt,:] = torch.tensor([Fb])
            len_D_ARR.append(len_Darray)
            cnt += 1

    return In, Out, len_D_ARR

# function to generate LSE dataset
def data_multi_paths(n_curves, b_min, b_max, b_grow, D_min_range, D_max_scale, len_Darray_range, approx_log_sum_flr, io_scale):
    b_array = createBetaArray(b_min, b_max, b_grow)
    n_beta_per_curve = len(b_array)
    n_data = n_curves * n_beta_per_curve
    In = torch.zeros((n_data, 12), dtype=torch.float32)
    Out = torch.zeros((n_data,1), dtype=torch.float32)
    len_D_ARR = []

    cnt = 0
    for c in range(n_curves):
        
        len_Darray = int(np.random.uniform(len_Darray_range[0], len_Darray_range[1]))
        D0 = np.random.uniform(D_min_range[0], D_min_range[1])
        D_array = np.random.uniform(D0, D_max_scale, len_Darray)
        D_10 = np.sort(D_array)[0:10]
        if approx_log_sum_flr:
            F_min = area_approx_F(D_10[0], D_max_scale, len_Darray, b_min)
        else:
            F_min = logSumExp(D_array, b_min)
        F_min = io_scale * F_min
        F_max = io_scale * D_10[0]

        for b in b_array:
            Fb = io_scale * logSumExp(D_array, b)
            In[cnt,:] = torch.tensor(
                np.concatenate((np.array([F_min]), D_10, np.array([myLog(b,10)])), axis=0)
                )
            # In[cnt,:] = torch.tensor([])
            # In[cnt,:] = torch.tensor([F_min, myLog(b,10)])
            Out[cnt,:] = torch.tensor([Fb])
            len_D_ARR.append(len_Darray)
            cnt += 1

    return In, Out, len_D_ARR


def visualizeData(In, Out, n_end):

    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.plot(In[:n_end,0], label='F_min')
    plt.legend()
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(In[:n_end,1], label='F_max')
    plt.legend()
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(In[:n_end,2], label='log beta')
    plt.legend()
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(In[:n_end,0], label='F_min', linestyle='dashed', linewidth=2)
    plt.plot(In[:n_end,1], label='F_max', linestyle='dashed', linewidth=2)
    plt.plot(Out[:n_end], label='F_beta')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12,2))
    plt.hist(In[:,1].numpy().flatten(), bins=10, density=True, histtype='barstacked', rwidth=0.8, label='d_min')
    plt.hist(In[:,0].numpy().flatten(), bins=10, density=True, histtype='barstacked', rwidth=0.8, label='F_min')
    plt.legend()

    plt.show()


