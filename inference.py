import torch
from BeamSearch import beam_search

def inference(data , model, method = 'BeamSearch'):

    num_data , num_cities , _ = data.shape
    device = data.device
    
    if method == 'Greedy':
        actions = torch.zeros(num_data, num_cities, 1).to(device)
        for t in range(1, num_cities):
            E = model.encoder(data)
            prev_chosen_indices = actions[:,t-1,0].unsqueeze(1).long()
            _ , outs = model.decoder(E , prev_chosen_indices, relevance_mask = None)
            actions[:, t, 0] = torch.argmax(outs, dim=-1)
            
    elif method == 'BeamSearch':
        seq, scores = beam_search(model,data,beam_width=min(num_cities,3))
        actions = seq[:,0,:]
    else:
        raise "Wrong Method!"
    
    return actions