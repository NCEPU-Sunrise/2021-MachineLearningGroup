import torch
import torch.nn as nn
from sublayers import EncoderLayer, DecoderLayer, PositionEncoding
from Transformer.utils.masking import get_mask


class Encoder(nn.Module):
    def __init__(self, src_vocab, d_k, d_v, idx, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, n_layers=6,
                 scale_emb=True, n_position=200):
        super(Encoder, self).__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=idx)
        self.positionEncoding = PositionEncoding(d_model, n_position)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.encoders = nn.ModuleList([
            EncoderLayer(d_k, d_v, d_model, n_heads, dropout, d_ff)
            for _ in range(n_layers)])

    def forward(self, src, attn_mask):
        enc_attn_list = []
        enc_input = self.src_embed(src)
        if self.scale_emb:
            enc_input = enc_input * self.d_model ** 0.5
        enc_output = self.positionEncoding(enc_input)
        for layer in self.encoders:
            enc_output, attn = layer(enc_output, attn_mask)
            enc_attn_list.append(attn)

        return enc_output, enc_attn_list


class Decoder(nn.Module):
    def __init__(self, trg_vocab, d_k, d_v, idx, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, n_layers=6,
                 scale_emb=True, n_position=200):
        super(Decoder, self).__init__()
        self.trg_embed = nn.Embedding(trg_vocab, d_model, padding_idx=idx)
        self.positionEncoding = PositionEncoding(d_model, n_position)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.decoders = nn.ModuleList([
            DecoderLayer(d_k, d_v, d_model, n_heads, dropout, d_ff)
            for _ in range(n_layers)])

    def forward(self, trg, enc_output, trg_mask, enc_dec_attn_mask):
        enc_dec_attn_list = []
        dec_attn_list = []
        dec_out = self.trg_embed(trg)
        if self.scale_emb:
            dec_out *= self.d_model ** 0.5
        dec_out = self.positionEncoding(dec_out)
        for layer in self.decoders:
            dec_out, dec_attn, enc_dec_attn = layer(dec_out, trg_mask, enc_output, enc_dec_attn_mask)
            dec_attn_list.append(dec_attn)
            enc_dec_attn_list.append(enc_dec_attn)
        return dec_out, dec_attn_list, enc_dec_attn_list


