from tqdm import tqdm
import torch
from torch import optim
from utils import *
from datetime import datetime
from torch_optimizer import RAdam



def train(model, data_loader, writer, lr = 0.001, len_print = 100, check_params = False):
    device = model.device
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = RAdam(model.parameters(), lr=lr)
    print(" Training Started with POMO.")
    batch_means = torch.zeros(len_print)
    best_mean_cost = float('inf')  # Initialize with infinity
    best_model_state = None
    torch.autograd.set_detect_anomaly(True)
    for episode, data_batch in tqdm(enumerate(data_loader)):
        baseline_rep_num = 8;
        data_batch = data_batch.to(device) # each data batch is batch_size number of different set of cities
        batch_size = data_batch.shape[0]
        expanded_data_batch = data_batch.unsqueeze(1).expand(-1, baseline_rep_num, -1, -1)  # Shape (batch_size, baseline_rep_num, N, d)
        expanded_data_batch = expanded_data_batch.reshape(batch_size * baseline_rep_num, *data_batch.shape[1:])  # Flatten for model input
        

        outs, actions = model(expanded_data_batch, mod = 'train')
        # print_gpu_memory_combined()
        with torch.no_grad():
            costs = route_cost(expanded_data_batch, actions)
        sum_log_prob = torch.log(outs.gather(2, actions.long()).squeeze(-1)).sum(dim=1) #log(p_theta)
        costs_m = costs.view(batch_size, baseline_rep_num).mean(dim=1, keepdim=True)
        baselines = costs_m.expand(-1,baseline_rep_num).reshape(batch_size*baseline_rep_num)  # Shape (batch_size, 1)
        policy_loss = torch.sum(sum_log_prob * (costs - baselines)) / batch_size #POMO Learning
        
    
        
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        batch_means[episode % len_print] = costs.mean().item()

        if check_params:
              check_model_weights(model)
              check_gradients(model)
        
        if episode % len_print == len_print - 1:
            
            mean_cost = batch_means.mean().item()
            print(f"Episode: {episode+1} Mean cost: {mean_cost:.2f}")
            writer.add_scalar('Mean cost', mean_cost, episode)
                  
            if mean_cost < best_mean_cost:
                best_mean_cost = mean_cost
                best_model_state = model.state_dict()
    
    if best_model_state is not None:
        torch.save(best_model_state, 'Saved_models/' + 'POMO' + datetime.now().strftime(("%Y_%m_%d %H_%M_%S")) + str(mean_cost) +'best_model.pth')
        print(f"Best model saved with mean cost: {best_mean_cost:.2f}")
    writer.close()
    return model

if __name__ == "__main__":
    print("test")