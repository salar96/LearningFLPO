import torch
import torch.nn as nn
import math
# torch.set_default_dtype(torch.float32)

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
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        # Compute sine and cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and move to the specified device
        pe = pe.unsqueeze(0).to(device)  # Shape: (1, max_len, d_model)
        # print('peeeeeee',pe.dtype)
        # Register as a buffer (non-trainable but saved with the model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :] * 0.1 # Slice to match sequence length
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, device, dropout=0.3, use_PE=True):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.positional_encoding = PositionalEncoding(hidden_dim, device)
        self.use_PE = use_PE  # Flag to control positional encoding usage
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4, # feedforward consists of two layers, and the last layer brings the dim back to hidden_dim
            dropout=dropout,
            batch_first=True,
            device=device
            # norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.fc1(x) #shape: (batch_size,num_cities,hidden_dim)
        if self.use_PE:
            x = self.positional_encoding(x)  # Add positional encoding if use_PE is True
        out = self.transformer_encoder(x).float()
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, device, dropout=0.3):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            device=device
            # norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.transformer_decoder = self.transformer_decoder.float() 
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """ 1. tgt (Target Sequence)
            Description : The input sequence to the decoder (queries in self-attention).
            Parallel to Tutorial : In the tutorial, this corresponds to the "target sequence" Y during training or the previously generated tokens Yt−1
            during inference.
            Shape : (batch_size, target_sequence_length, hidden_dim).
            2. memory (Encoder Output)
            Description : The encoded representation of the input sequence from the encoder (keys and values in cross-attention).
            Parallel to Tutorial : This corresponds to X_hat the output of the encoder.
            Shape : (batch_size, input_sequence_length, hidden_dim). 
            3. tgt_mask (Target Mask)
            Description : A mask applied to the target sequence to enforce causality (prevent attending to future tokens).
            Parallel to Tutorial : This corresponds to the "causal mask" mentioned in the tutorial. It ensures that each position in the target sequence can only attend to itself and prior positions.
            Shape : (target_sequence_length, target_sequence_length).
            4. memory_mask (Memory Mask)
            Description : A mask applied to the encoder output (memory). This is rarely used in practice but can be helpful in certain scenarios (e.g., masking out irrelevant parts of the input).
            Parallel to Tutorial : Not explicitly discussed in the tutorial, but it would correspond to any additional masking applied to the encoder output.
            5. tgt_key_padding_mask (Target Padding Mask)
            Description : A mask to ignore padding tokens in the target sequence.
            Parallel to Tutorial : This corresponds to ignoring padded tokens in the target sequence (if applicable). For example, if the target sequence is shorter than the maximum length, padding tokens are ignored.
            Shape : (batch_size, target_sequence_length). 
            6. memory_key_padding_mask (Memory Padding Mask)
            Description : A mask to ignore padding tokens in the encoder output (memory).
            Parallel to Tutorial : This corresponds to ignoring padded tokens in the input sequence (if applicable). For example, if the input sequence is shorter than the maximum length, padding tokens are ignored.
            Shape : (batch_size, input_sequence_length)."""
    
        out = self.transformer_decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True
        ) # returns the transformed values of tgt after cross-attention
        return out

class VRPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_enc_layers=3, num_dec_layers=3, num_heads=8, dropout=0.3, use_PE=False):
        super(VRPNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, num_enc_layers, num_heads, device, dropout, use_PE)
        self.decoder = Decoder(hidden_dim, num_dec_layers, num_heads, device, dropout)
        self.pe = PositionalEncoding(hidden_dim, device)
        self.init_args = {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'device': device, 
                          'num_enc_layers': num_enc_layers, 'num_dec_layers': num_dec_layers, 
                          'num_heads': num_heads, 'dropout': dropout, 'use_PE': use_PE}

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, X, mod='train'):
        batch_size, seq_length, _ = X.size()
        encoded_cities = self.encoder(X)
        outs = torch.zeros(batch_size, seq_length, seq_length).to(self.device)
        action_indices = torch.zeros(batch_size, seq_length, 1).to(self.device)
        indices_to_ignore = None
        end_selected = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for t in range(seq_length):
            # Prepare masks and target sequence
            if t == 0:
                # Force first action to be START (index 0)
                memory_key_padding_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=self.device)
                memory_key_padding_mask[:, 0] = False  # Unmask START
                tgt = self.pe(encoded_cities[:, 0:1, :])  # Use START city embedding
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
            else:
                # Select previously chosen cities
                prev_chosen_cities = action_indices[:, :t, 0].long()
                selected_cities = encoded_cities[torch.arange(batch_size).unsqueeze(1), prev_chosen_cities, :]
                tgt = self.pe(selected_cities)
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)

                # Create memory mask: block visited cities and handle END depot
                memory_key_padding_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=self.device)
                for i in range(batch_size):
                    if end_selected[i]:
                        # Mask all except END depot
                        memory_key_padding_mask[i, :seq_length-1] = True
                    else:
                        # Mask visited cities (except END)
                        memory_key_padding_mask[i, indices_to_ignore[i]] = True

            # Decode and compute attention scores
            dec_out = self.decoder(
                tgt=tgt,
                memory=encoded_cities,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            query = dec_out[:, -1, :]
            keys = encoded_cities
            scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            scores = scores.squeeze(1)

            # Apply masks
            if t == 0:
                # Ensure only START is allowed at t=0
                scores_mask = torch.ones_like(scores, dtype=torch.bool)
                scores_mask[:, 0] = False
                scores = scores.masked_fill(scores_mask, float('-inf'))
            if memory_key_padding_mask is not None:
                scores = scores.masked_fill(memory_key_padding_mask, float('-inf'))

            # Compute probabilities and select next city
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.clamp(attn_weights, min=1e-9)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

            if mod == 'train':
                idx = torch.multinomial(attn_weights, num_samples=1).squeeze(-1)
            elif mod == 'eval_greedy':
                idx = torch.argmax(attn_weights, dim=-1)
            else:
                raise ValueError("Invalid mode")

            # Force END depot selection once triggered
            idx = torch.where(end_selected, torch.full_like(idx, seq_length - 1), idx)

            # Update end_selected flag
            with torch.no_grad():
                end_selected = end_selected | (idx == (seq_length - 1))

            # Update outputs and tracking
            outs[:, t, :] = attn_weights
            action_indices[:, t, 0] = idx

            # Track visited cities
            if t == 0:
                indices_to_ignore = idx.unsqueeze(-1)
            else:
                indices_to_ignore = torch.cat((indices_to_ignore, idx.unsqueeze(-1)), dim=-1).long()

        outs = torch.clamp(outs, min=1e-9, max=1.0)
        return outs, action_indices