import torch
from BeamSearch import beam_search
from RelevanceMaskGenerator import RelevanceMaskGenerator
def inference(data , model, method = 'BeamSearch'):

    num_data , num_cities , _ = data.shape
    device = data.device
    P_s = None
    if method == 'Greedy':
        actions = torch.zeros(num_data, num_cities,1).to(device)
        E = model.encoder(data)
        depo = data[:, -1, :] 
        for t in range(1, num_cities):

            indices = actions[:,t-1,0].long()
            stokens = data[
                torch.arange(num_data).unsqueeze(1), indices.unsqueeze(1), :
                ]
            with torch.no_grad():
                m1 = torch.zeros(num_data, num_cities).to(device)
                m1[torch.arange(num_data).unsqueeze(1), actions[:,:t,0].long()] = 1
                end_selected = indices == (
                num_cities - 1
                )  # shows if we are already at the depot
                m1[end_selected, :-1] = (
                    1  # mask all except last node (depot) for batches at depot
                )
                # m2 = RelevanceMaskGenerator(data, indices).float()
                # relevance_mask = ((m1 + m2) > 0).float()
                relevance_mask = m1
                relevance_mask[:, -1] = 0  # Last node (depot) should not be selected
            _ , outs = model.decoder(E , stokens.squeeze(1), depo, mask = relevance_mask)
            actions[:, t, 0] = torch.argmax(outs, dim=-1)

    elif method == 'BeamSearch':
        beam_width = min(5, num_cities - 1)  # Ensure beam width does not exceed number of cities
        actions, scores = beam_search(model, data, beam_width=beam_width)
    elif method == "sampling":
        actions = torch.zeros(num_data, num_cities, 1).to(device)
        P_s = torch.zeros(num_data, num_cities, num_cities).to(device)
        E = model.encoder(data)
        depo = data[:, -1, :]
        for t in range(1, num_cities):

            indices = actions[:,t-1,0].long()
            stokens = data[
                torch.arange(num_data).unsqueeze(1), indices.unsqueeze(1), :
                ]
            with torch.no_grad():
                m1 = torch.zeros(num_data, num_cities).to(device)
                m1[torch.arange(num_data).unsqueeze(1), actions[:,:t,0].long()] = 1
                end_selected = indices == (
                num_cities - 1
                )  # shows if we are already at the depot
                m1[end_selected, :-1] = (
                    1  # mask all except last node (depot) for batches at depot
                )
                # m2 = RelevanceMaskGenerator(data, indices).float()
                # relevance_mask = ((m1 + m2) > 0).float()
                relevance_mask = m1
                relevance_mask[:, -1] = 0  # Last node (depot) should not be selected
            _ , outs = model.decoder(E , stokens.squeeze(1), depo, mask = relevance_mask)
            with torch.no_grad():
                a  = torch.multinomial(outs, num_samples=1).squeeze(-1)
                actions[:, t, 0] = a
            P_s[:, t, :] = outs
    elif method == "free_sampling":
        actions = torch.zeros(num_data, num_cities, 1).to(device)
        P_s = torch.ones(num_data, num_cities, num_cities).to(device)
        P_s[:, :, 0] = 0.0 # coming back to start is prohibited
        P_s[:, -1, :-1] = 0.0 # only destination to destination
        P_s = P_s/torch.sum(P_s, axis=-1, keepdims=True) # normalization
        for t in range(1, num_cities):
            a = torch.multinomial(P_s[:, t, :], num_samples=1).squeeze(-1)
            actions[:, t, 0] = a
    else:
        raise ValueError("Wrong Method!")

    return P_s , actions