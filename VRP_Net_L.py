import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=500):
        """
        By using (1, max_len, d_model), the positional encoding can be added directly to the input embeddings
        of shape (batch_size, seq_len, d_model) without requiring additional reshaping or computation.
        PyTorch automatically broadcasts the 1 in the batch dimension to match the batch size of the input.
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding with explicit dtype=torch.float32
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # Compute sine and cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and move to the specified device
        pe = pe.unsqueeze(0).to(device)  # Shape: (1, max_len, d_model)
        # Register as a buffer (non-trainable but saved with the model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, : x.size(1), :]  # Slice to match sequence length
        return x


class LinearAttnEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        nlayers=3,
        dim_feedforward=128,
        dropout=0.3,
        num_query_vectors=16,
    ):
        super().__init__()

        dim_multiplier = 2

        self.attn1_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    dim_feedforward * dim_multiplier,
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
                    dim_feedforward * dim_multiplier,
                    nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(nlayers)
            ]
        )
        # Learnable Key and Value matrices K,V (m × d)
        self.key_vectors = nn.Parameter(
            torch.randn(num_query_vectors, dim_feedforward * dim_multiplier)
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model + 1, dim_feedforward * dim_multiplier)
        self.linear2 = nn.Linear(dim_feedforward * dim_multiplier, dim_feedforward)

        self.norm = nn.LayerNorm(dim_feedforward * dim_multiplier)
        # self.norm2 = nn.LayerNorm(dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.nlayers = nlayers

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: (batch_size, n, d) → Input sequence X
        """

        batch_size, n_, d_ = src.shape

        roles = torch.zeros(batch_size, n_, 1).to(src.device)
        roles[:,-1,0] += 1

        src = torch.cat([src,roles], dim=-1)
        src = self.dropout(self.activation(self.linear1(src)))

        for i in range(self.nlayers):

            I, _ = self.attn1_layers[i](
                query=self.key_vectors.unsqueeze(0).expand(batch_size, -1, -1),
                key=src,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )

            # assert not torch.isnan(I).any(), f"NaN in I before attn2, layer {i}"
            attn_output, _ = self.attn2_layers[i](
                query=src,
                key=I,
                value=I,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            # assert not torch.isnan(attn_output).any(), f"NaN in attn_output, layer {i}"
            src = self.norm(src + self.dropout(attn_output))

        src = self.linear2(src)

        return src  # Output shape: (batch_size, n, d)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, device, dropout=0.3):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            device=device,
            # norm_first=True
        )
        self.pe = PositionalEncoding(hidden_dim, device)
        self.hidden_dim = hidden_dim
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.transformer_decoder = self.transformer_decoder.float()
        self.action_indices = None
        self.indices_to_ignore = None  # keep track of visited nodes
        
        self.end_selected = None

    def generate_square_subsequent_mask(
        self, sz
    ):  # enforcing causaliy of decoder sequence
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(
        self,
        E,
        prev_chosen_indices,
        relevance_mask = None
    ):
        
        batch_size, seq_length, _ = E.size()
            
        selected_cities = E[
                torch.arange(batch_size).unsqueeze(1), prev_chosen_indices, :
            ]
        
        tgt = self.pe(selected_cities)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        # print(prev_chosen_indices)
        end_selected = (prev_chosen_indices[: , -1] == (seq_length - 1))
        # print(end_selected)
        memory_key_padding_mask = torch.zeros(
            (batch_size, seq_length), dtype=torch.bool, device=E.device
        )
        for i in range(batch_size):

            if end_selected[i]:
                # Mask all except END depot
                memory_key_padding_mask[i, : seq_length - 1] = True
            else:
                # Mask visited cities (except END)
                memory_key_padding_mask[i, prev_chosen_indices[i]] = True
        
        dec_out = self.transformer_decoder(
            tgt=tgt,
            memory=E,
            tgt_mask=tgt_mask,
        )  # returns the transformed values of tgt after cross-attention
        
        query = torch.mean(dec_out, dim=1, keepdim=True)
        
        scores = torch.bmm(query, E.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        scores = scores.squeeze(1) # (B , n)
        
        
        visited_cities_penalty = memory_key_padding_mask.float() * -1e6
        scores += visited_cities_penalty
        if relevance_mask is not None:
            scores += relevance_mask * -1.0

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.clamp(attn_weights, min=1e-9)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
       
        return scores, attn_weights


class VRPNet_L(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        device,
        num_enc_layers=3,
        num_dec_layers=3,
        num_heads=8,
        dropout=0.3,
        use_PE=False,
    ):
        super(VRPNet_L, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = LinearAttnEncoder(
            input_dim,
            num_heads,
            num_enc_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        ).to(device)

        self.decoder = Decoder(hidden_dim, num_dec_layers, num_heads, device, dropout)
        
        
        
        self.init_args = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "device": device,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "use_PE": use_PE,
        }

        print('Model created.')



if __name__ == "__main__":

    data = torch.rand(10,5,2)
    model = VRPNet_L(2,64,'cpu',1,1,8,0.3,False)
    E = model.encoder(data)
    prev_chosen_indices = (torch.zeros(10)+3).unsqueeze(-1).long()
    p = model.decoder(E,prev_chosen_indices)
    print(p)
