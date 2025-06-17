import numpy as np
import torch
import pickle

# def generate_dataset(n, cov, k, n_clusters,  scale, seed):
#     # Generate random means for each cluster
#     np.random.seed(seed)
#     means = np.random.rand(n_clusters, 2)*scale

#     # Generate random points in k clusters
#     nodes = np.zeros((n, 2))
#     for i in range(n):
#         cluster = np.random.randint(0, means.shape[0])  # Choose a random cluster
#         nodes[i] = np.random.multivariate_normal(means[cluster], cov)
#     Y_s = np.tile(np.sum(nodes, axis=0), (k,1))/n + np.random.rand(k, 2)*scale*0.1
#     dest = np.array([[0.9,0.8]])*scale
#     return nodes,Y_s,dest

def generate_dataset(
    num_drones, 
    n_drone_clusters, 
    drone_cluster_split,
    num_facilities, 
    dim_, 
    device,
    drone_cluster_std_range = [0.01, 0.05], 
    F_noise_std = 0.005,
    F_noise_mean = 0.0,
    num_distinct_ends = 1
):
    # Assign start location to each drone
    drone_cnt = 0
    for i in range(n_drone_clusters):
        if i == n_drone_clusters - 1:
            n_drones = int(num_drones - drone_cnt)
        else:
            n_drones = int(drone_cluster_split[i] * (num_drones + 1))
            drone_cnt += n_drones

        drone_cluster_mean = (
            torch.rand(1, dim_).repeat(n_drones, 1).unsqueeze(1).to(device)
        )
        drone_cluster_std = (
            ((drone_cluster_std_range[0] - drone_cluster_std_range[1]) * torch.rand(1, dim_) + drone_cluster_std_range[1])
            .repeat(n_drones, 1)
            .unsqueeze(1)
            .to(device)
        )
        drone_cluster_START_locs = torch.normal(
            mean=drone_cluster_mean, std=drone_cluster_std
        ).to(device)
        if i == 0:
            START_locs = drone_cluster_START_locs
        else:
            START_locs = torch.cat((START_locs, drone_cluster_START_locs), axis=0)

    assert not START_locs.requires_grad, "set requires_grad for START_locs to 0"
    # Assign destination location to each drone
    num_ends = num_drones // num_distinct_ends
    remaining_ends = num_drones - num_ends*num_distinct_ends
    
    END_locs = torch.rand(1,1,dim_, requires_grad=False, device=device).expand(num_ends,-1,-1)
    for _ in range(num_distinct_ends-1):
        END_locs = torch.cat((END_locs,torch.rand(1,1,dim_, requires_grad=False, device=device).expand(num_ends,-1,-1)),dim=0)
    END_locs = torch.cat((END_locs,torch.rand(1,1,dim_, requires_grad=False, device=device).expand(remaining_ends,-1,-1)),dim=0)
    # END_locs = 0.5*torch.ones(
    #     (num_drones, 1, dim_), requires_grad=False, device=device
    # )  # torch.rand(num_drones, 1, dim_, requires_grad=False, device=device)

    # Create the data tensor
    F_means = torch.mean(START_locs, dim=0).repeat(num_facilities, 1)
    F_noise = torch.normal(
        mean = F_noise_mean * torch.ones(num_facilities, dim_, device=device),
        std = F_noise_std * torch.ones(num_facilities, dim_, device=device),
    )
    F_base = (F_means + F_noise).unsqueeze(0).requires_grad_()

    assert F_base.requires_grad == True
    assert START_locs.requires_grad == False
    assert END_locs.requires_grad == False

    print("Data Created.")

    return START_locs, F_base, END_locs

def torchFLPO_2_numpyFLPO(START_locs, END_locs, F_base, file_dir, scale):

    node_locations = START_locs.cpu().numpy().squeeze()
    destination_location = END_locs[0].cpu().numpy()
    facility_location = F_base.detach().cpu().numpy().squeeze()

    numpyFLPOdata = {}
    numpyFLPOdata['nodeLocations'] = node_locations
    numpyFLPOdata['destinationLocation'] = destination_location
    numpyFLPOdata['facilityLocations'] = facility_location
    numpyFLPOdata['numFacilities'] = facility_location.shape[0]
    numpyFLPOdata['numNodes'] = node_locations.shape[0]
    numpyFLPOdata['scale'] = scale

    with open(file_dir, 'wb') as file:
        pickle.dump(numpyFLPOdata, file)

