import torch
import torch.nn as nn
from layers import Encoder, Decoder
from Transformer.utils.masking import get_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, src_idx, trg_idx, d_model=512,
                 n_heads=8, d_ff=2048, d_k=64, d_v=64, dropout=0.1,
                 scale_embed=True, n_layers=6, n_position=200, scale_prj=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, d_k, d_v, src_idx, d_model,
                               n_heads, dropout, d_ff, n_layers, scale_embed, n_position)
        self.decoder = Decoder(trg_vocab, d_k, d_v, trg_idx, d_model,
                               n_heads, dropout, d_ff, n_layers, scale_embed, n_position)
        self.src_idx = src_idx
        self.trg_idx = trg_idx
        self.d_model = d_model
        self.projection = nn.Linear(d_model, trg_vocab, bias=False)
        self.scale_prj = scale_prj
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg):
        src_mask = None
        trg_mask = None

        enc_output, *_ = self.encoder(src, src_mask)
        dec_output, *_ = self.decoder(trg, enc_output, trg_mask, src_mask)
        seq_logit = self.projection(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        return seq_logit.view(-1, seq_logit.shape[-1])



