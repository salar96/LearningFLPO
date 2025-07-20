import torch
import torch.nn as nn


def beam_search(

    model: nn.Module,
    data: torch.Tensor,
    beam_width: int = 3,
    start_idx: int = 0,
    device: torch.device = None,
):
 
    """
    Performs beam search decoding to find the best sequence of cities for a routing problem.
    Args:
        model (nn.Module): Neural network model with encoder and decoder components.
        data (torch.Tensor): Input tensor of shape (batch_size, num_nodes, feature_dim) containing
                            coordinates of cities/nodes to visit.
        beam_width (int, optional): Number of beams to keep track of at each step. Defaults to 3.
        start_idx (int, optional): Index of the starting node. Defaults to 0.
        device (torch.device, optional): Device to run computations on. If None, uses data's device.
                                        Defaults to None.
    Returns:
        tuple:
            - sequences (torch.Tensor): Tensor of shape (batch_size, beam_width, seq_length) containing
                                       the top-k sequences for each batch.
            - scores (torch.Tensor): Tensor of shape (batch_size, beam_width) containing the
                                    log-probability scores for each sequence.
    Note:
        The function assumes the model has an encoder and decoder component.
        The encoder processes the input data to create a memory representation.
        The decoder uses this memory and previous selections to predict the next node to visit.
        The function keeps track of visited nodes to ensure each node is visited exactly once.
        The last node in data is assumed to be the depot.
    """
    model.eval()
    B, N, _ = data.size()
    if device is None:
        device = data.device

    data = data.to(device)
    with torch.no_grad():
        memory = model.encoder(data)  # (B, N, d)

    # Initial state
    sequences = torch.full(
        (B, 1, 1), start_idx, dtype=torch.long, device=device
    )  # (B, 1, 1)
    scores = torch.zeros(B, 1, device=device)
    visited = torch.zeros(B, 1, N, dtype=torch.bool, device=device)
    visited[:, :, start_idx] = True  # mark start as visited

    depot = data[:, -1, :]  # (B, d)

    for t in range(1, N):
        beam_size = sequences.size(1)

        flat_seqs = sequences.view(B * beam_size, -1)  # (B*beam, t)
        prev_city_idx = flat_seqs[:, -1]  # (B*beam,)
        # Get previous chosen cities' coordinates
        data_exp = (
            data.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(B * beam_size, N, -1)
        )
        prev_city = torch.gather(
            data_exp,
            1,
            prev_city_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, data_exp.size(2)),
        ).squeeze(
            1
        )  # (B*beam, d)

        memory_exp = (
            memory.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(B * beam_size, N, -1)
        )
        depot_exp = (
            depot.unsqueeze(1).expand(-1, beam_size, -1).reshape(B * beam_size, -1)
        )  # (B*beam, d)

        # Visited mask: (B*beam, N)
        visited_flat = visited.view(B * beam_size, N)

        end_selected = prev_city_idx == (N - 1)
        visited_flat[end_selected, :-1] = True  # mask all except last node (depot)

        mask = visited_flat

        _, logits = model.decoder(memory_exp, prev_city, depot_exp, mask=mask)
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
    from SPN import SPN
    from utils import route_cost, load_model
    from inference import inference
    from matplotlib import pyplot as plt
    from pathlib import Path
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    data = torch.rand(1, 50, 2).to(device)
    
    model_classes = {"SPN": SPN}
    weights_address = Path("Saved_models") / "SPN100FastDecoderBest.pth"
    spn = load_model(weights_address, model_classes, weights_only=True, device=device)
    for param in spn.parameters():
        param.requires_grad = False
    print("SPN loaded on: ", spn.device)
    beam_search_result, beam_scores = beam_search(
        spn, data, beam_width=3, start_idx=0, device=device
    )
    
    # now compare with greedy search
    _, greedy_actions = inference(data, spn, method="Greedy")
    print("Greedy Actions:", greedy_actions)
    cost_greedy = route_cost(data, greedy_actions)
    print(f"Greedy Cost: {cost_greedy}")

    print("Beam Search Actions:")
    for i in range(beam_search_result.size(1)):
        beam_actions = beam_search_result[:, i, :].unsqueeze(-1)
        cost_beam = route_cost(data, beam_actions)
        # print(f"Beam {i} Actions: {beam_actions}")
        print(f"Beam {i} Cost: {cost_beam}")
        print(f"Beam {i} Score: {beam_scores[:, i]}")
