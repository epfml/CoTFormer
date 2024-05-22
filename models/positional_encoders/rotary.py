import torch
from torch import nn

from .encoder import PositionalEncoder, PositionalEncoderClosure
from .rotary_utils import apply_rotary_emb


class RotaryPositionalEncoderClosure(PositionalEncoderClosure):

    def adapt_vector_for_indices(self, v, indices):
        #changer = torch.zeros_like(indices)
        #changer[50::51] = 1
        #indices -= torch.cumsum(changer, dim=-1)

        *other_dims, T, hs = v.shape
        if T == 0:
            return v
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        other_dims_prefix = other_dims[:len(other_dims) - len(indices.shape) + 1]
        freqs = (indices.unsqueeze(-1) * self.encoder.freqs.view(1, 1, -1)).unsqueeze(-1).expand(*indices.shape, -1, 2).reshape(*indices.shape, hs)
        freqs = freqs.view([indices.shape[0]] + [1] * len(other_dims_prefix) + list(indices.shape[1:]) + [hs]).expand(*v.shape)
        v = apply_rotary_emb(freqs, v)
        return v

    def _adapt_keys_for_indices(self, k, indices):
        return self.adapt_vector_for_indices(k, indices)

    def _adapt_queries_for_indices(self, q, indices):
        return self.adapt_vector_for_indices(q, indices)


class RotaryPositionalEncoder(PositionalEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.max_pos_log = 4
        self.max_pos_base = 10  
        n_embd_per_head = config.n_embd // config.n_head
        freqs =  (self.max_pos_base ** (-self.max_pos_log * torch.arange(0, n_embd_per_head, 2)[:(n_embd_per_head // 2)].float() / n_embd_per_head))
        self.register_buffer("freqs", freqs)

    closure_model = RotaryPositionalEncoderClosure
