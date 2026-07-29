"""Plain PyTorch transformer encoder, built entirely from config (no downloads, no checkpoints).

Pre-norm, classic recipe. Positional encoding is sinusoidal (no learned params) so the
parameter count stays exactly analyzable: 12*d^2*L (attention + FFN) plus a small embedding
term, per "Training: The Four Line Items of GPU Memory".
"""
import math

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ff_mult, seq_len, use_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "pos_encoding", _sinusoidal_position_encoding(seq_len, d_model), persistent=False
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * ff_mult,
                    batch_first=True,
                    norm_first=True,  # pre-norm
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        x = self.embed(tokens) + self.pos_encoding[:seq_len].unsqueeze(0)
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Recompute activations on the backward pass instead of storing them.
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(x)


def _sinusoidal_position_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analytical_param_count(d_model, n_layers, vocab_size, ff_mult=4):
    """12*d^2*L at ff_mult=4 (4d^2 attention + 2*ff_mult*d^2 FFN), plus embedding + output head.

    Ignores biases and layernorm params (~13*d*L total) -- a few percent at most, and the
    contract this benchmark tests explicitly treats them as negligible.
    """
    attn_and_ffn = (d_model**2) * (4 + 2 * ff_mult) * n_layers
    embeddings = 2 * vocab_size * d_model  # input embedding + output head, untied
    return attn_and_ffn + embeddings
