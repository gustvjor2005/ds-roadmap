import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, in_features=5, d_model=32, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        print(f"\nDebug - Input shape: {x.shape}")
        
        # Project input to d_model dimensions
        x = self.input_proj(x)
        print(f"After projection shape: {x.shape}")
        
        # Self attention - Each sequence attends to itself only
        # Q, K, V are all the same input tensor
        attn_out, attn_weights = self.self_attention(x, x, x)
        print(f"Attention output shape: {attn_out.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")  # Shows how each position attends to others
        
        # First residual connection and layer norm
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward - Processes each position independently
        ff_out = self.ff(x)
        print(f"Feedforward output shape: {ff_out.shape}")
        
        # Second residual connection and layer norm
        x = self.norm2(x + self.dropout(ff_out))
        
        # Take only the last sequence element for prediction
        x = x[:, -1, :]
        print(f"After selecting last token: {x.shape}")
        
        # Final projection
        x = self.final(x)
        print(f"Final output shape: {x.shape}")
        return x.squeeze(-1)