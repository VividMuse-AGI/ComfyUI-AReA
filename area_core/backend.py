"""Attention without diffusers; tensor layout is [batch, sequence, heads, dim]."""

import torch.nn.functional as F


def dispatch_attention_fn(query, key, value, attn_mask=None, backend=None):
    if backend is not None:
        return backend(query, key, value, attn_mask)
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        attn_mask=attn_mask,
    ).transpose(1, 2)
