import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from RelevanceMaskGenerator import RelevanceMaskGenerator

def beam_search(model: nn.Module,
                data: torch.Tensor,
                beam_width: int = 3,
                start_idx: int = 0,
                device: torch.device = None):
    """
    Batched beam search with visited city tracking and relevance masking.
    """
    model.eval()
    B, N, _ = data.size()
    if device is None:
        device = data.device

    data = data.to(device)
    with torch.no_grad():
        memory = model.encoder(data)  # (B, N, d)

    # Initial state
    sequences = torch.full((B, 1, 1), start_idx, dtype=torch.long, device=device)  # (B, 1, 1)
    scores = torch.zeros(B, 1, device=device)
    visited = torch.zeros(B, 1, N, dtype=torch.bool, device=device)
    visited[:, :, start_idx] = True  # mark start as visited

    for t in range(1, N):
        beam_size = sequences.size(1)

        flat_seqs = sequences.view(B * beam_size, -1)  # (B*beam, t)
        prev_city = flat_seqs[:, -1].unsqueeze(1)  # (B*beam, 1)

        memory_exp = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(B * beam_size, N, -1)
        data_exp = data.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(B * beam_size, N, -1)

        # Visited mask: (B*beam, N)
        visited_flat = visited.view(B * beam_size, N)

        # Relevance mask: (B*beam, N)
        m2 = RelevanceMaskGenerator(data_exp, prev_city)
        mask = visited_flat | m2  # Combined mask

        _, logits = model.decoder(memory_exp, prev_city, relevance_mask=mask)
        log_probs = torch.log(logits + 1e-9)

        log_probs = log_probs.view(B, beam_size, N)
        total_scores = scores.unsqueeze(-1) + log_probs  # (B, beam, N)

        total_scores_flat = total_scores.view(B, -1)
        topk_scores, topk_indices = torch.topk(total_scores_flat, beam_width, dim=-1)

        beam_idx = topk_indices // N
        city_idx = topk_indices % N

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        selected_seqs = sequences[batch_idx, beam_idx]
        new_seqs = torch.cat([selected_seqs, city_idx.unsqueeze(-1)], dim=-1)

        sequences = new_seqs
        scores = topk_scores

        selected_visited = visited[batch_idx, beam_idx]  # (B, beam, N)
        updated_visited = selected_visited.clone()
        updated_visited.scatter_(2, city_idx.unsqueeze(-1), True)
        visited = updated_visited

    return sequences, scores


if __name__ == "__main__":
    from VRP_Net_L import VRPNet_L
    from utils import route_cost, load_model
    from inference import inference
    from matplotlib import pyplot as plt
    from pathlib import Path
    seed=1;
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cpu')
    print("Running on: " , device)
    data = torch.rand(1,50,2)
    torch.cuda.empty_cache()
    model_classes = {"VRPNet_L": VRPNet_L}
    weights_address = (
        Path("Saved_models") /
        "VRPNet_L_lr1e-04_bs32_ep60000_samples1920000_cities50_inputdim2_"
        "workers0_hidden64_enc1_dec1_heads8_dropout0.30_"
        "train_PO_2025_05_17_22_43_32last_model.pth"
    )
    vrp_net = load_model(
        weights_address, model_classes, weights_only=True, device=device
    )
    seq, scores = beam_search(vrp_net,data,beam_width=10)
 
    [print(route_cost(data, seq[:,i,:])) for i in range(seq.shape[1])]
    
    # def plot_routes(cities, routes):
    #     for batch_index in torch.arange(len(cities)):
    #         # Extract the specific batch
    #         cities_batch = cities[batch_index].numpy()
    #         route_batch = routes[batch_index].long().squeeze().numpy()
    #         # Get coordinates of cities in the order of the route
    #         ordered_cities = cities_batch[route_batch]
    #         # Plot cities
    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(
    #             cities_batch[1:-1, 0],
    #             cities_batch[1:-1, 1],
    #             marker=".",
    #             color="blue",
    #             zorder=2,
    #             label="Cities",
    #         )
     
    #         plt.plot(
    #             ordered_cities[:, 0],
    #             ordered_cities[:, 1],
    #             color="red",
    #             linestyle="-",
    #             zorder=1,
    #             label="Route",
    #         )

    #         # Highlight start and end points
    #         plt.scatter(
    #             cities_batch[0, 0],
    #             cities_batch[0, 1],
    #             color="green",
    #             s=50,
    #             label="Start",
    #             zorder=3,
    #         )
    #         plt.scatter(
    #             cities_batch[-1, 0],
    #             cities_batch[-1, 1],
    #             marker="^",
    #             color="red",
    #             s=50,
    #             label="End",
    #             zorder=3,
    #         )
    #         model_cost = route_cost(
    #             cities[batch_index : batch_index + 1], routes[batch_index : batch_index + 1]
    #         )[
    #             0
    #         ]  # /straight_costs(cities[batch_index:batch_index+1])[0]
            
    #         plt.title(
    #             f"Batch {batch_index} cost: {model_cost:.2f}"
    #         )

    #         plt.xlabel("X Coordinate")
    #         plt.ylabel("Y Coordinate")
    #         plt.axis("off")
    #         plt.show()
    #         # Save the plot to the specified folder
    # plot_routes(data, actions)
            
       



    
