import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically slice pos_embed up to sequence length of x
        seq_len = x.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pos_embed.size(1)}")
        return x + self.pos_embed[:, :seq_len, :]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        d_model: int = 512, 
        n_heads: int = 8, 
        n_layers: int = 4,
        ff_dim: int = 2048, 
        dropout: float = 0.1, 
        max_spatial: int = 256
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # +1 for the CLS token
        self.pos_enc = LearnablePositionalEncoding(max_spatial + 1, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)
        
        # Project channel dimension to d_model
        x = self.proj(x)  # (B, H*W, d_model)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, H*W + 1, d_model)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Return only the CLS token
        return x[:, 0, :]
