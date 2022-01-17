import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        q: [B, H, L, dk]
        k: [B, H, L, dk]
        v: [B, H, L, dv]
        """
        d_k = k.shape[-1]
        attn = torch.matmul(q, k.transpose(2, 3)) / d_k ** 0.5
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads=8, d_model=512, dropout=0.1):
        """Default that d_k == d_v == d_model / n_heads = 64"""
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_ks = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_vs = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.attention = ScaledDotProductAttention(dropout)
        self.projection = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, attn_mask=None):
        """
        q -> [B, H, L, dk]
        k -> [B, H, L, dk]
        v -> [B, H, L, dv]
        output -> [B, L, H * dv]
        """
        batch_size, length, n_heads = q.shape[0], q.shape[1], self.n_heads
        residual = q
        q = self.w_qs(q).view(batch_size, length, n_heads, -1)
        v = self.w_vs(v).view(batch_size, length, n_heads, -1)
        k = self.w_ks(k).view(batch_size, length, n_heads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output, attn = self.attention(q, k, v, attn_mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, length, -1)
        output = self.dropout(self.projection(output))
        output += residual
        output = self.layernorm(output)
        return output, attn














