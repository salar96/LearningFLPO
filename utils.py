import math
import torch
from scipy.stats import ttest_rel
import subprocess
import numpy as np
from scipy.spatial.distance import cdist
import uavFLPO as UFO


class UAV_Net:
    def __init__(self, drones, num_stations, distance='sqeuclidean') -> None:
        super().__init__()
        
        self.drones = drones
        self.N_drones= len(drones)
        self.N_stations=num_stations
        self.stage_horizon=self.N_stations+1
        self.gamma_k_length=self.N_stations+1
        self.distance=distance
        
        return

    def return_stagewise_cost(self,params_drones): #params is like stations
        my_inf = 1e6;
        
        D_ss=[0]*self.N_drones
        for drone_id, _ in enumerate(self.drones):
            params = params_drones[drone_id]
            d_F=cdist(params,params,self.distance)
            d_F=d_F+np.diag([my_inf]*self.N_stations)
            d_delta_to_f=np.array([my_inf]*self.N_stations).reshape(1,-1)
            d_df=np.concatenate((d_F,d_delta_to_f),axis=0)
            stage=np.concatenate((params,np.array(self.drones[drone_id][1]).reshape(1,-1)),axis=0)
            D_s=[0]*(self.stage_horizon+1)
            stage_0=np.array(self.drones[drone_id][0]).reshape(1,-1)
            D_s[0]=cdist(stage_0,stage,self.distance)
            d_f_to_delta=cdist(params,np.array(self.drones[drone_id][1]).reshape(1,-1),self.distance)
            d_last=np.concatenate((d_f_to_delta,np.array([0]).reshape(1,-1)),axis=0)
            d=np.concatenate((d_df,d_last),axis=1)
            D_s[1:self.stage_horizon] = [d] * (self.stage_horizon - 1)
            d_l=[my_inf]*(self.gamma_k_length-1)
            d_l.append(0.0)
            D_s[-1]=np.array(d_l).reshape(-1,1)
            D_ss[drone_id]=D_s
        
        return D_ss
    
def calc_associations(D_ss,beta):
        """ This function calculates the association probabilities over all drones for all stages.
        The input D_ss is a list, each element is another list, where each element of the latter is cost matrix 
        from one stage to another.
        beta is the temperature parameter. """
        p=[]
    
        for D_s in D_ss:
            K=len(D_s)
            D=D_s[::-1]
            out_D=[0]*(K+1)
            out_D[0]=np.array([0.0]).reshape(-1,1)
            out_p=[0]*(K+1)
            out_p[0]=np.array([1.0]).reshape(-1,1)
            out=[0]*(K+1)
            out[0]=np.array([1.0]).reshape(-1,1)
            for i in range(1,K+1):
                out_D[i]=(D[i-1]+np.repeat(np.transpose(out_D[i-1]),D[i-1].shape[0],axis=0))
                m=out_D[i].min(axis=1,keepdims=True)
                exp_D=np.exp(np.multiply(-beta,out_D[i]-m))
                out[i]=np.sum(np.multiply(exp_D,np.tile(out[i-1], (1,D[i-1].shape[0])).T),axis=1,keepdims=True)
                out_p[i]=np.divide(np.multiply(exp_D,out[i-1].T),out[i])
                out_D[i]=m
            p.append(out_p[::-1][:-1])
        
        return p

def calc_routs(P_ss):
        """ Given the associations for each drone, return the routs (facility ids) for each. """
        O=[]
        A=[]
        for i in range(len(P_ss)):
          indices = [0]
          first_action = np.zeros(len(P_ss[i]))
          first_action[0]=1
          o=[first_action]
          a = [0]
          for p in P_ss[i]:
              m = np.argmax(p[indices[-1],:])
              a.append(m+1)
              o.append(np.concatenate(([0],p[indices[-1],:])))
              indices.append(m)
          o.pop()
          a.pop()
          O.append(o)
          A.append(a)
        return np.array(O, dtype=object),np.array(A, dtype=object) # recently changed A to np.array(A)

def generate_true_labels(data_batch, beta):
    num_facilities = data_batch.shape[1]-2
    drones = torch.cat((data_batch[:,0:1,:], data_batch[:,-1:,:]), dim=1).detach().cpu().numpy()
    uav_net = UAV_Net(drones, num_facilities)
    params = data_batch[:,1:-1,:].detach().cpu().numpy()
    P_ss = calc_associations(uav_net.return_stagewise_cost(params),beta = beta)
    # print(P_ss[0])
    label_outs , label_actions = calc_routs(P_ss)
    return label_outs , label_actions

def generate_true_labels1(data_batch, beta):
    num_facilities = data_batch.shape[1]-2
    drones = torch.cat((data_batch[:,0:1,:], data_batch[:,-1:,:]), dim=1).detach().cpu().numpy()
    # calculate associations for each drone
    P_ss = []
    params_batch = data_batch[:,1:-1,:].detach().cpu().numpy()
    # print(f'params_batch:{params_batch}')
    for i, drone in enumerate(drones):
        start_loc = drone[0].reshape(-1,2)
        end_loc = drone[1].reshape(-1,2)
        flpo_agent = UFO.FLPO(start_loc, end_loc, num_facilities, scale=1.0, disType='sqeuclidean', selfHop=False)
        params = params_batch[i]
        # print(f'params:{params}')
        D_s, _ = flpo_agent.returnStagewiseCost(params)
        _, _, _, _, P_s = flpo_agent.backPropDP(D_s, beta=beta, returnPb=True)
        P_ss.append(P_s)
    # calculate routs and action labels
    label_outs , label_actions = calc_routs(P_ss)
    return label_outs, label_actions

def print_gpu_memory_combined():
    # PyTorch memory stats
    total_memory = (
        torch.cuda.get_device_properties(0).total_memory / 1e6
    )  # Total GPU memory in MB
    allocated_memory = torch.cuda.memory_allocated() / 1e6  # Allocated memory in MB
    reserved_memory = torch.cuda.memory_reserved() / 1e6  # Reserved memory in MB
    free_memory_pytorch = total_memory - (
        allocated_memory + reserved_memory
    )  # Available memory in MB

    # NVIDIA-SMI memory stats
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    total, used, free = map(int, result.stdout.strip().split(","))

    print("PyTorch Memory Stats:")
    print(f"  Total Memory: {total_memory:.2f} MB")
    print(f"  Allocated Memory: {allocated_memory:.2f} MB")
    print(f"  Reserved Memory: {reserved_memory:.2f} MB")
    print(f"  Available Memory (PyTorch): {free_memory_pytorch:.2f} MB")

    print("\nNVIDIA-SMI Memory Stats:")
    print(f"  Total Memory: {total} MB")
    print(f"  Used Memory: {used} MB")
    print(f"  Free Memory: {free} MB")


def route_cost(cities, routes):
    B, N, _ = cities.shape
    routes = routes.squeeze(-1).long()  # Convert to long for indexing
    ordered_cities = cities[
        torch.arange(B).unsqueeze(1), routes
    ]  # Reorder cities based on routes
    diffs = (
        ordered_cities[:, :-1] - ordered_cities[:, 1:]
    )  # Compute differences between consecutive cities
    distances = torch.norm(diffs, p=2, dim=2) ** 2  # Euclidean distances
    total_distances = distances.sum(dim=1)  # Sum distances for each batch
    return total_distances


def num_flpo_routes(n_facilities, n_drones):
    n_int_stages = n_facilities + 1
    n_routes_flip = [[]] * n_int_stages

    for i in range(n_int_stages):
        if i == 0:
            n_routes_flip[i] = [1] * (n_facilities + 1)
        elif i > 0 and i < n_int_stages - 1:
            n_routes_flip[i] = [sum(n_routes_flip[i - 1])] * n_facilities
            n_routes_flip[i].append(1)  # only one path from stopping state
        elif i == n_int_stages - 1:
            n_routes_flip[i] = [sum(n_routes_flip[i - 1])] * n_drones

    return n_routes_flip[::-1]


def load_model(path, model_classes, args=None, device="cuda", weights_only=True):
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    model_class_name = checkpoint["model_class"]
    init_args = checkpoint["init_args"]
    state_dict = checkpoint["state_dict"]

    # Inject the right device
    init_args["device"] = device

    model_class = model_classes[model_class_name]
    if args is None:
        model = model_class(**init_args)
    else:
        model = model_class(**args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_unit_circle_cities(B, N, d):
    """
    Generates a PyTorch tensor of size (B, N, d), representing B batches
    of N cities in d-dimensional space, where cities are randomly placed on the unit circle.

    Args:
        B (int): Number of batches.
        N (int): Number of cities in each batch.
        d (int): Number of dimensions (must be at least 2, higher dimensions will have zeros).

    Returns:
        torch.Tensor: A tensor of shape (B, N, d) with cities on the unit circle.
    """
    if d < 2:
        raise ValueError("Dimension 'd' must be at least 2.")

    # Generate random angles for each city
    angles = torch.rand(B, N) * 2 * math.pi  # Random angles in radians

    # Coordinates on the unit circle
    x_coords = torch.cos(angles)
    y_coords = torch.sin(angles)

    # Create a tensor of zeros for higher dimensions if d > 2
    higher_dims = torch.zeros(B, N, d - 2)

    # Combine x, y, and higher dimensions
    unit_circle_coords = torch.stack((x_coords, y_coords), dim=-1)
    result = torch.cat((unit_circle_coords, higher_dims), dim=-1)
    shuffled_indices = torch.stack([torch.randperm(N) for _ in range(B)])
    shuffled_indices = shuffled_indices.unsqueeze(-1).expand(-1, -1, d)
    shuffled_x = torch.gather(result, dim=1, index=shuffled_indices)
    # result[:,0,:] = result[:,-1,:]
    return shuffled_x


def check_model_weights(model, large_value_threshold=1e6):
    for name, param in model.named_parameters():
        # Check for NaN
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")

        # Check for large values
        if torch.any(torch.abs(param) > large_value_threshold):
            print(
                f"Large value detected in parameter: {name}, max value: {param.abs().max().item()}"
            )


def check_gradients(model, large_grad_threshold=1e6):
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check for NaN in gradients
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradient of parameter: {name}")

            # Check for large gradient values
            if torch.any(torch.abs(param.grad) > large_grad_threshold):
                print(
                    f"Large gradient detected in parameter: {name}, max gradient: {param.grad.abs().max().item()}"
                )


def createBetaArray(b_min, b_max, grow):
    beta = b_min
    beta_array = [beta]
    while beta < b_max:
        beta = beta * grow
        beta_array.append(beta)
    beta_array = torch.tensor(beta_array, dtype=torch.float32)
    return beta_array


def logSumExp(D_tensor, beta):
    # with torch.no_grad():
    # print(D_tensor)
    D_min = torch.min(D_tensor, axis=1, keepdims=True)
    F = (
        -1
        / beta
        * torch.log(
            torch.sum(
                torch.exp(-beta * (D_tensor - D_min.values)), axis=1, keepdims=True
            )
        )
        + 1 / beta * torch.log(torch.tensor([D_tensor.shape[1]]))
        + D_min.values
    )
    return F


def area_approx_F(D_min, D_max_range, beta, printCalculations=False):
    min_beta_D_arr = beta * D_min
    x_max = beta * D_max_range - min_beta_D_arr
    F_est = -1 / beta * torch.log(1 / x_max * (1 - torch.exp(-x_max))) + D_min

    if printCalculations:
        print(f"min_beta_D_arr:{min_beta_D_arr}")
        print(f"x_max:{x_max}")
        print(f"inside_log:{1/x_max * (1 - torch.exp(-x_max))}")
        print(f"log:{torch.log(1/x_max * (1 - torch.exp(-x_max)))}")
        print(f"-1/beta_log:\n{-1/beta * torch.log(1/x_max * (1 - torch.exp(-x_max)))}")

    return F_est


coordinates = [
    [6734, 1453],
    [2233, 10],
    [5530, 1424],
    [401, 841],
    [3082, 1644],
    [7608, 4458],
    [7573, 3716],
    [7265, 1268],
    [6898, 1885],
    [1112, 2049],
    [5468, 2606],
    [5989, 2873],
    [4706, 2674],
    [4612, 2035],
    [6347, 2683],
    [6107, 669],
    [7611, 5184],
    [7462, 3590],
    [7732, 4723],
    [5900, 3561],
    [4483, 3369],
    [6101, 1110],
    [5199, 2182],
    [1633, 2809],
    [4307, 2322],
    [675, 1006],
    [7555, 4819],
    [7541, 3981],
    [3177, 756],
    [7352, 4506],
    [7545, 2801],
    [3245, 3305],
    [6426, 3173],
    [4608, 1198],
    [23, 2216],
    [7248, 3779],
    [7762, 4595],
    [7392, 2244],
    [3484, 2829],
    [6271, 2135],
    [4985, 140],
    [1916, 1569],
    [7280, 4899],
    [7509, 3239],
    [10, 2676],
    [6807, 2993],
    [5185, 3258],
    [3023, 1942],
]

# Convert the list to a PyTorch tensor
USA_data = torch.tensor(coordinates, dtype=torch.float).unsqueeze(0)
# 7762 is the max
USA_data = USA_data / torch.max(USA_data)
