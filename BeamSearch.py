import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def beam_search(model: nn.Module,
                data: torch.Tensor,
                beam_width: int = 3,
                start_idx: int = 0,
                device: torch.device = None):
    """
    Efficient beam search decoding for a Transformer-like model.

    Args:
        model (nn.Module): Model with encoder and decoder methods.
        data (torch.Tensor): Input tensor of shape (B, N, d).
        beam_width (int): Beam size.
        start_idx (int): Start token index.
        device (torch.device): Computation device.

    Returns:
        torch.LongTensor: Shape (B, beam_width, N), decoded sequences.
        torch.FloatTensor: Shape (B, beam_width), log-prob scores.
    """
    model.eval()
    B, N, _ = data.size()
    if device is None:
        device = data.device

    with torch.no_grad():
        E = model.encoder(data.to(device))  # (B, N, d)

    all_seqs = []
    all_scores = []

    for b in range(B):
        E_b = E[b:b+1]  # (1, N, d)

        beams = [([start_idx], 0.0)]  # (sequence list, cumulative log-prob)

        for _ in range(N - 1):
            beam_seqs = [torch.tensor(seq, dtype=torch.long) for seq, _ in beams]
            beam_input = pad_sequence(beam_seqs, batch_first=True).to(device)  # (beam_width, t)

            # Expand encoder memory for each beam
            E_b_exp = E_b.expand(len(beams), -1, -1)  # (beam_width, N, d)
            _, probs = model.decoder(E_b_exp, beam_input)  # (beam_width, N)
            log_probs = torch.log(probs)  # (beam_width, N)

            candidates = []
            for i, (seq, score) in enumerate(beams):
                topk_lp, topk_idx = torch.topk(log_probs[i], beam_width)
                for lp, idx in zip(topk_lp, topk_idx):
                    new_seq = seq + [idx.item()]
                    new_score = score + lp.item()
                    candidates.append((new_seq, new_score))

            # Prune to top beams
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Final beam results for this batch
        seqs = [seq for seq, _ in beams]
        scores = [score for _, score in beams]
        all_seqs.append(seqs)
        all_scores.append(scores)

    seq_tensor = torch.tensor(all_seqs, dtype=torch.long, device=device)  # (B, beam_width, N)
    score_tensor = torch.tensor(all_scores, dtype=torch.float, device=device)  # (B, beam_width)

    return seq_tensor, score_tensor


if __name__ == "__main__":
    from VRP_Net_L import VRPNet_L
    data = torch.rand(10,5,2)
    model = VRPNet_L(2,64,'cpu',1,1,8,0.3,False)
    seq, scores = beam_search(model,data,beam_width=5)
    print(scores)
    
