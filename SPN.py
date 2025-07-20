import torch
import torch.nn as nn
import math


class LinearAttnEncoder(nn.Module):
    """
    A custom encoder module utilizing multi-head attention layers and learnable key vectors.
    Args:
        dInput (int): Dimension of the input features (excluding the role indicator).
        nhead (int): Number of attention heads in each MultiheadAttention layer.
        nlayers (int, optional): Number of stacked attention layers. Default is 3.
        dModel (int, optional): Dimension of the model (hidden size). Default is 128.
        dropout (float, optional): Dropout probability for attention and feedforward layers. Default is 0.3.
        num_query_vectors (int, optional): Number of learnable key vectors for the induced attention block. Default is 16.
    Attributes:
        attn1_layers (nn.ModuleList): List of MultiheadAttention layers for the first attention block.
        attn2_layers (nn.ModuleList): List of MultiheadAttention layers for the second attention block.
        key_vectors (nn.Parameter): Learnable key vectors used as queries in the first attention block.
        LinearIN (nn.Linear): Linear layer to project input features (plus role indicator) to model dimension.
        AttnFFlayers (nn.ModuleList): List of feedforward networks applied after attention blocks.
        norm1 (nn.LayerNorm): Layer normalization applied after the second attention block.
        norm2 (nn.LayerNorm): Layer normalization applied after the feedforward network.
        dropout (nn.Dropout): Dropout layer.
        activation (nn.ReLU): Activation function.
        nlayers (int): Number of stacked layers.
    Forward Args:
        src (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dInput).
        src_mask (torch.Tensor, optional): Attention mask for the attention layers.
        src_key_padding_mask (torch.Tensor, optional): Padding mask for the attention layers.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, sequence_length, dModel) after encoding.
    """

    def __init__(
        self,
        dInput,
        nhead,
        nlayers=3,
        dModel=128,
        dropout=0.3,
        num_query_vectors=16,
    ):
        super().__init__()

        dim_multiplier = 2

        self.attn1_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    dModel,
                    nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(nlayers)
            ]
        )
        self.attn2_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    dModel,
                    nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(nlayers)
            ]
        )
        # Learnable Key and Value matrices K,V (m × d)
        self.key_vectors = nn.Parameter(torch.randn(num_query_vectors, dModel))

        # Feedforward network
        self.LinearIN = nn.Linear(dInput + 1, dModel)

        self.AttnFFlayers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dModel, dModel * dim_multiplier),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dModel * dim_multiplier, dModel),
                    nn.Dropout(dropout),
                )
                for _ in range(nlayers)
            ]
        )

        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.nlayers = nlayers

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        batch_size, n_, d_ = src.shape

        roles = torch.zeros(batch_size, n_, 1).to(src.device)
        roles[:, -1, 0] += 1

        src = torch.cat([src, roles], dim=-1)
        src = self.dropout(self.activation(self.LinearIN(src)))

        for i in range(self.nlayers):

            I, _ = self.attn1_layers[i](
                query=self.key_vectors.unsqueeze(0).expand(batch_size, -1, -1),
                key=src,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )

            attn_output, _ = self.attn2_layers[i](
                query=src,
                key=I,
                value=I,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = self.norm1(src + self.dropout(attn_output))
            ff_out = self.AttnFFlayers[i](src)
            src = self.norm2(src + ff_out)

        return src


class Decoder(nn.Module):
    """
    A PyTorch module implementing a gated multi-head attention decoder for sequence modeling tasks.
    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden representations.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate for multi-head attention layers. Default is 0.3.
    Attributes:
        Linear_s (nn.Linear): Linear layer to project the current state.
        Linear_d (nn.Linear): Linear layer to project the depot state.
        MHA_s (nn.MultiheadAttention): Multi-head attention for the current state.
        MHA_d (nn.MultiheadAttention): Multi-head attention for the depot state.
        gating_vector (nn.Parameter): Learnable parameter for gating mechanism.
    Forward Args:
        E (torch.Tensor): Encoded input sequence of shape (batch_size, seq_length, input_dim).
        s (torch.Tensor): Current state tensor of shape (batch_size, input_dim).
        d (torch.Tensor): Depot state tensor of shape (batch_size, input_dim).
        mask (torch.Tensor, optional): Mask tensor for attention scores, shape (batch_size, seq_length).
    Returns:
        scores (torch.Tensor): Raw attention scores of shape (batch_size, seq_length).
        attn_weights (torch.Tensor): Normalized attention weights of shape (batch_size, seq_length).
    Notes:
        - The decoder computes gated attention between the current and depot states over the encoded sequence.
        - The gating mechanism blends the two attention outputs using a learnable vector.
        - Masking is applied to attention scores to prevent attending to certain positions.
    """

    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.3):
        super(Decoder, self).__init__()

        self.Linear_s = nn.Linear(input_dim, hidden_dim)  # project current state
        self.Linear_d = nn.Linear(input_dim, hidden_dim)  # project depot state
        self.MHA_s = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.MHA_d = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.hidden_dim = hidden_dim
        self.gating_vector = nn.Parameter(torch.randn(2 * hidden_dim))
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, E, s, d, mask=None):

        S = self.norm(self.act(self.Linear_s(s))).unsqueeze(1)  # (B, 1, hidden_dim)
        D = self.norm(self.act(self.Linear_d(d))).unsqueeze(1)  # (B, 1, hidden_dim)
        h_s, _ = self.MHA_s(S, E, E)  # (B, 1, hidden_dim)
        h_d, _ = self.MHA_d(D, E, E)  # (B, 1, hidden_dim)

        concat = torch.cat([h_s, h_d], dim=-1)  # (B, 1, 2*hidden_dim)
        alpha = torch.sigmoid(
            torch.matmul(concat, self.gating_vector.unsqueeze(-1))
        )  # (B, 1, 1)

        query = alpha * h_s + (1 - alpha) * h_d

        scores = torch.bmm(query, E.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        scores = scores.squeeze(1)  # (B , n)

        if mask is not None:
            scores += mask.float() * -1e6

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.clamp(attn_weights, min=1e-9)

        return scores, attn_weights



class FastDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.3):
        super(FastDecoder, self).__init__()

        self.Linear_s = nn.Linear(input_dim, hidden_dim)  # project current state
        self.Linear_d = nn.Linear(input_dim, hidden_dim)  # project depot state
        self.gating_vector = nn.Parameter(torch.randn(2 * hidden_dim))
        self.hidden_dim = hidden_dim
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, E, s, d, mask=None):
        """
        E: (B, n, dModel) — encoded node embeddings
        s: (B, dInput)   — current state
        d: (B, dInput)   — depot state
        mask: (B, n) optional masking
        """

        z_s = self.norm(self.act(self.Linear_s(s))).unsqueeze(1) # (B, 1, dModel)
        z_d = self.norm(self.act(self.Linear_d(d))).unsqueeze(1)  # (B, 1, dModel)

        concat = torch.cat([z_s, z_d], dim=-1)  # (B, 1, 2*dModel)
        alpha = torch.sigmoid(torch.matmul(concat, self.gating_vector.unsqueeze(-1)))  # (B, 1, 1)

        query = alpha * z_s + (1 - alpha) * z_d  # (B, 1, dModel)

        scores = torch.bmm(query, E.transpose(1, 2)) / math.sqrt(self.hidden_dim)  # (B, 1, n)
        scores = scores.squeeze(1)  # (B, n)

        if mask is not None:
            scores += mask.float() * -1e6

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.clamp(attn_weights, min=1e-9)

        return scores, attn_weights


class SPN(nn.Module):
    """
    Shortest Path Network (SPN) module.
    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        num_enc_layers (int, optional): Number of encoder layers. Default is 3.
        num_heads (int, optional): Number of attention heads. Default is 8.
        dropout (float, optional): Dropout rate. Default is 0.3.
    Attributes:
        hidden_dim (int): Dimension of hidden layers.
        device (torch.device): Device used for computation.
        encoder (LinearAttnEncoder): Encoder module for input features.
        decoder (Decoder): Decoder module for output generation.
        init_args (dict): Dictionary of initialization arguments.
    Note:
        This class implements a neural network architecture for shortest path problems,
        consisting of an encoder and decoder with attention mechanisms.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        device,
        num_enc_layers=3,
        num_heads=8,
        dropout=0.3,
        fast_decoder=True,
    ):
        super(SPN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = LinearAttnEncoder(
            input_dim,
            num_heads,
            num_enc_layers,
            hidden_dim,
            dropout=dropout,
        ).to(device)
        if fast_decoder:
            self.decoder = FastDecoder(input_dim, hidden_dim, num_heads, dropout).to(device)
        else:
            self.decoder = Decoder(input_dim, hidden_dim, num_heads, dropout).to(device)
        self.init_args = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "device": device,
            "num_enc_layers": num_enc_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "fast_decoder": fast_decoder,
        }

        print("Model created.")


if __name__ == "__main__":
    from SPN import SPN

    # Example usage
    input_dim = 2  # Example input dimension
    hidden_dim = 64  # Example hidden dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SPN(
        input_dim, hidden_dim, device, num_enc_layers=3, num_heads=8, dropout=0.3
    )
    print(model)
    
