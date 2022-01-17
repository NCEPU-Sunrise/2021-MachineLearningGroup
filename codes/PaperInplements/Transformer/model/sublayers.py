import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        res = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(self.linear2(x)) + res
        x = self.layernorm(x)
        return x


class PositionEncoding(nn.Module):
    def __init__(self, d_model, n_pos=200):
        super(PositionEncoding, self).__init__()
        self.PE = torch.zeros(1, n_pos, d_model)
        pos = torch.arange(0, n_pos, dtype=torch.float32).reshape(-1, 1)
        div = torch.arange(0, d_model, step=2, dtype=torch.float32) / d_model
        div = -div * np.log(10000.0)
        pos = torch.exp(pos + div)
        self.PE[:, :, 0::2] = torch.sin(pos)
        self.PE[:, :, 1::2] = torch.cos(pos)

    def forward(self, x):
        x = x + self.PE[:, :x.shape[1], :].to(x.device)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model=512, n_heads=8, dropout=0.1, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_k, d_v, n_heads, d_model, dropout)
        self.FFN = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, enc_input, enc_attn_mask):
        enc_output, enc_attn = self.attention(enc_input, enc_input, enc_input, enc_attn_mask)
        out = enc_input + enc_output
        out = self.layernorm(out)
        out = out + self.FFN(out)
        return self.layernorm(out), enc_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model=512, n_heads=8, dropout=0.1, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.dec_attention = MultiHeadAttention(d_k, d_v, n_heads, d_model, dropout)
        self.enc_dec_attntion = MultiHeadAttention(d_model, d_model, n_heads, d_model, dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.FFN = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, dec_input, dec_attn_mask, enc_output, enc_dec_attn_mask):
        dec_out, dec_attn = self.dec_attention(dec_input, dec_input, dec_input, dec_attn_mask)
        dec_out = self.layernorm(dec_out + dec_input)
        enc_dec_out, enc_dec_attn = self.enc_dec_attntion(dec_out, enc_output, enc_output, enc_dec_attn_mask)
        out = self.layernorm(enc_dec_out + dec_out)
        out = self.layernorm(out + self.FFN(out))
        return out, dec_attn, enc_dec_attn


