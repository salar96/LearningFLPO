import math
import torch
from scipy.stats import ttest_rel
import subprocess

def is_paired_ttest_significant(tensor1, tensor2, alpha=0.05):
    """
    Perform a one-sided paired t-test to check if tensor1 is significantly smaller than tensor2.

    Args:
        tensor1 (torch.Tensor): 1D tensor of values.
        tensor2 (torch.Tensor): 1D tensor of values (same length as tensor1).
        alpha (float): Significance level (default is 0.05).

    Returns:
        bool: True if the p-value for the one-sided test is below alpha, False otherwise.
    """
    # Ensure inputs are 1D tensors of the same length
    if tensor1.ndim != 1 or tensor2.ndim != 1:
        raise ValueError("Both tensors must be 1-dimensional.")
    if len(tensor1) != len(tensor2):
        raise ValueError("Both tensors must have the same length.")

    # Convert tensors to numpy arrays
    array1 = tensor1.cpu().numpy()
    array2 = tensor2.cpu().numpy()

    # Perform the paired t-test
    t_stat, p_value_two_sided = ttest_rel(array1, array2)

    # Adjust for one-sided test (assuming we test if tensor1 < tensor2)
    if t_stat > 0:
        # If the mean of tensor1 is greater than tensor2, the one-sided p-value is 1 - p/2
        p_value_one_sided = 1 - (p_value_two_sided / 2)
    else:
        # If the mean of tensor1 is smaller than tensor2, the one-sided p-value is p/2
        p_value_one_sided = p_value_two_sided / 2

    # Check if the one-sided p-value is below the significance level
    return p_value_one_sided < alpha



def print_gpu_memory_combined():
    # PyTorch memory stats
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e6  # Total GPU memory in MB
    allocated_memory = torch.cuda.memory_allocated() / 1e6  # Allocated memory in MB
    reserved_memory = torch.cuda.memory_reserved() / 1e6  # Reserved memory in MB
    free_memory_pytorch = total_memory - (allocated_memory + reserved_memory)  # Available memory in MB

    # NVIDIA-SMI memory stats
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        text=True
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
    ordered_cities = cities[torch.arange(B).unsqueeze(1), routes]  # Reorder cities based on routes
    diffs = ordered_cities[:, :-1] - ordered_cities[:, 1:]  # Compute differences between consecutive cities
    distances = torch.norm(diffs, p=2, dim=2)**2  # Euclidean distances
    total_distances = distances.sum(dim=1)  # Sum distances for each batch
    return total_distances

def num_flpo_routes(n_facilities, n_drones):
    n_int_stages = n_facilities + 1
    n_routes_flip = [[]]*n_int_stages

    for i in range(n_int_stages):
        if i == 0:
            n_routes_flip[i] = [1]*(n_facilities+1)
        elif i > 0 and i < n_int_stages-1:
            n_routes_flip[i] = [sum(n_routes_flip[i-1])] * n_facilities
            n_routes_flip[i].append(1) # only one path from stopping state
        elif i == n_int_stages - 1:
            n_routes_flip[i] = [sum(n_routes_flip[i-1])] * n_drones

    return n_routes_flip[::-1]


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
    result[:,0,:] = result[:,-1,:]
    return result

def check_model_weights(model, large_value_threshold=1e6):
    for name, param in model.named_parameters():
        # Check for NaN
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")
        
        # Check for large values
        if torch.any(torch.abs(param) > large_value_threshold):
            print(f"Large value detected in parameter: {name}, max value: {param.abs().max().item()}")

def check_gradients(model, large_grad_threshold=1e6):
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check for NaN in gradients
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradient of parameter: {name}")
            
            # Check for large gradient values
            if torch.any(torch.abs(param.grad) > large_grad_threshold):
                print(f"Large gradient detected in parameter: {name}, max gradient: {param.grad.abs().max().item()}")


def createBetaArray(min, max, grow):
    beta = min
    beta_array = [beta]
    while beta <= max:
        beta = beta*grow
        beta_array.append(beta)
    beta_array = torch.tensor(beta_array, dtype=torch.float32)
    return beta_array


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
    [3023, 1942]
]

# Convert the list to a PyTorch tensor
USA_data = torch.tensor(coordinates, dtype=torch.float).unsqueeze(0)
# 7762 is the max
USA_data = USA_data / torch.max(USA_data)


    