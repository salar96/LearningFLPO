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
def data_multi_paths(n_curves, b_array, D_range, num_D_range, n_shortest, io_scale=1, plothist=False):
    # b_array = createBetaArray(b_min, b_max, b_grow)
    n_beta_per_curve = len(b_array)
    n_data = n_curves * n_beta_per_curve
    In = torch.zeros((n_data, n_shortest+2), dtype=torch.float32)
    Out = torch.zeros((n_data,1), dtype=torch.float32)
    len_D_ARR = []

    cnt = 0
    for c in range(n_curves):
        
        len_Darray = int(np.random.uniform(num_D_range[0], num_D_range[1]))
        n_intervals = np.random.choice(range(2,100))
        n_act_intervals = np.random.choice(range(1,n_intervals))
        distr = PiecewiseUniformDistribution(
            a=D_range[0], 
            b=D_range[1],
            num_intervals=n_intervals,
            num_active_intervals=n_act_intervals,
            seed=None)
        D_array = distr.sample(len_Darray)
        D_shortest = np.sort(D_array)[0:n_shortest]
        F_min = io_scale * logSumExp(D_array, b_array[0])
        F_max = io_scale * D_shortest[0]

        for b in b_array:
            Fb = io_scale * logSumExp(D_array, b)
            In[cnt,:] = torch.tensor(
                np.concatenate((np.array([F_min]), D_shortest, np.array([myLog(b,10)])), axis=0)
                )
            Out[cnt,:] = torch.tensor([Fb])
            len_D_ARR.append(len_Darray)
            cnt += 1

        # Plot histogram and PDF
        if plothist:
            plt.hist(D_array, bins=100, density=True, alpha=0.5, label='Sampled')
            xs = np.linspace(D_range[0], D_range[1], 1000)
            ys = [distr.pdf(x) for x in xs]
            plt.plot(xs, ys, label='PDF', color='red')
            plt.legend()
            plt.title("Piecewise Uniform Distribution")
            plt.show()

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


class PiecewiseUniformDistribution:
    def __init__(self, a, b, num_intervals, num_active_intervals, seed=None):
        """
        Initializes a piecewise uniform distribution on [a, b].

        Parameters:
            a, b: Interval endpoints (a < b)
            num_intervals: Total number of subintervals
            num_active_intervals: Number of subintervals with non-zero density
            seed: Random seed for reproducibility
        """
        assert a < b, "a must be less than b"
        assert num_active_intervals <= num_intervals, "Cannot have more active intervals than total intervals"

        self.a = a
        self.b = b
        self.num_intervals = num_intervals
        self.num_active_intervals = num_active_intervals
        self.seed = seed
        self._generate_intervals()

    def _generate_intervals(self):
        rng = np.random.default_rng(self.seed)
        self.interval_edges = np.linspace(self.a, self.b, self.num_intervals + 1)
        self.interval_lengths = np.diff(self.interval_edges)

        # Choose active intervals randomly
        all_indices = np.arange(self.num_intervals)
        self.active_indices = rng.choice(all_indices, size=self.num_active_intervals, replace=False)

        # Normalize to create a proper probability distribution
        active_lengths = self.interval_lengths[self.active_indices]
        total_length = np.sum(active_lengths)
        self.density = 1.0 / total_length

    def pdf(self, x):
        """Returns the PDF value at x."""
        if x < self.a or x > self.b:
            return 0.0
        idx = np.searchsorted(self.interval_edges, x, side='right') - 1
        if idx in self.active_indices:
            return self.density
        else:
            return 0.0

    def sample(self, N):
        """Draw N samples from the distribution."""
        rng = np.random.default_rng(self.seed)
        active_lengths = self.interval_lengths[self.active_indices]
        probs = active_lengths / np.sum(active_lengths)
        chosen_indices = rng.choice(self.active_indices, size=N, p=probs)
        samples = rng.uniform(
            self.interval_edges[chosen_indices],
            self.interval_edges[chosen_indices + 1]
        )
        return samples

