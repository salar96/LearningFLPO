
from utils import route_cost, generate_unit_circle_cities
from VRP_Net_L import VRPNet_L
from data_loader_script import *
from train import train
import torch
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchinfo import summary
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
mp.set_start_method('spawn', force=True)
torch.cuda.empty_cache()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="Running Neural TSP Learning")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
parser.add_argument('--use_base', 
                        type=str2bool, 
                        default=True, 
                        help="Enable using base (true/false). Default is True.")
args = parser.parse_args()

def preload_data(data_loader, device):
    preloaded_batches = []
    for data_batch in data_loader:
        preloaded_batches.append(data_batch.to(device, non_blocking=True))
    return preloaded_batches

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " , device)

    lr = 1e-3
    batch_size = 32
    num_episodes = 1000

    num_samples = batch_size * num_episodes
    num_cities = 50
    input_dim = 2
    num_workers = 0  #
    data_loader = create_data_loader(batch_size, num_samples, num_cities, input_dim, num_workers=num_workers)
#     eval_loader = create_evaluation_loader(
#     num_samples=1024,
#     num_cities=num_cities,
#     input_dim=input_dim,
#     batch_size=32,
#     num_workers=num_workers
# )
    run_name = 'VRP/' + str(batch_size) + '_' + str(num_cities) + '_' + str(num_samples) + '_' + '/ANN/'+datetime.now().strftime(("%Y_%m_%d %H_%M_%S")) + str(args.lr)
    writer = SummaryWriter(log_dir=run_name)


    hidden_dim = 32
    num_layers_enc = 1
    num_layers_dec = 1
    num_heads = 8

    model = VRPNet_L(input_dim, hidden_dim, device, num_layers_enc, num_layers_dec, num_heads)
    print(model.device)
    print(summary(model))
    
    trained = train(model, data_loader, writer, lr,len_print=10)


