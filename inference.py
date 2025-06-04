import torch
from BeamSearch import beam_search
from RelevanceMaskGenerator import RelevanceMaskGenerator
def inference(data , model, method = 'Greedy'):

    num_data , num_cities , _ = data.shape
    device = data.device
    P_s = None
    if method == 'Greedy':
        actions = torch.zeros(num_data, num_cities, 1).to(device)
        E = model.encoder(data)
        for t in range(1, num_cities):
            
            prev_chosen_indices = actions[:,t-1,0].unsqueeze(1).long()
            with torch.no_grad():
                m1 = torch.zeros(num_data, num_cities, dtype=bool).to(device)
                m1[torch.arange(num_data).unsqueeze(1), actions[:,:t,0].long()] = True
                m2 = RelevanceMaskGenerator(data, prev_chosen_indices)
                relevance_mask = m1 | m2
            _ , outs = model.decoder(E , prev_chosen_indices, relevance_mask = relevance_mask)
            actions[:, t, 0] = torch.argmax(outs, dim=-1)
            
    elif method == 'BeamSearch':
        seq, scores = beam_search(model,data,beam_width=3)
        actions = seq[:,0,:]
    elif method == "sampling":
        actions = torch.zeros(num_data, num_cities, 1).to(device)
        P_s = torch.zeros(num_data, num_cities, num_cities).to(device)
        E = model.encoder(data)
        for t in range(1, num_cities):
            
            prev_chosen_indices = actions[:,t-1,0].unsqueeze(1).long()
            with torch.no_grad():
                m1 = torch.zeros(num_data, num_cities, dtype=bool).to(device)
                m1[torch.arange(num_data).unsqueeze(1), actions[:,:t,0].long()] = True
                m2 = RelevanceMaskGenerator(data, prev_chosen_indices)
                relevance_mask = m1 | m2
            _ , outs = model.decoder(E , prev_chosen_indices, relevance_mask = relevance_mask)
            with torch.no_grad():
                a  = torch.multinomial(outs, num_samples=1).squeeze(-1)
                actions[:, t, 0] = a
            P_s[:, t, :] = outs
    else:
        raise "Wrong Method!"
    
    return P_s , actions