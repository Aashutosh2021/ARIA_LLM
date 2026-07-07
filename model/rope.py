"""
AIRA-LLM
Rotary Positional Embeddings (RoPE)
"""

import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for RoPE.
    Matches HuggingFace's rotate_half layout where freqs are repeated.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    return torch.stack([cos, sin], dim=0)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to input tensors using the precomputed cos/sin.
    freqs_cis has shape (2, seq_len, head_dim)
    xq and xk have shape (batch, seq_len, num_heads, head_dim)
    """
    cos = freqs_cis[0].to(xq.device)
    sin = freqs_cis[1].to(xq.device)
    
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

