import torch
from torch import nn


class PositionalEncoderClosure(object):

    def __init__(self, encoder):
        self.encoder = encoder

    def adapt_model_input(self, x, start_index=None, indices=None):
        return x

    def adapt_keys(self, k, start_index=None, indices=None):
        if indices is None:
            T = k.shape[-2]
            indices = torch.arange(start_index, T + start_index, device=k.device)
        return self._adapt_keys_for_indices(k, indices)

    def _adapt_keys_for_indices(self, k, indices):
        return k

    def _adapt_queries_for_indices(self, q, indices):
        return q

    def adapt_queries(self, q, start_index=None, indices=None):
        if indices is None:
            T = q.shape[-2]
            indices = torch.arange(start_index, T + start_index, device=q.device)
        return self._adapt_queries_for_indices(q, indices)

    def adapt_attention_before_softmax(self, att, start_query_index=None, start_key_index=None, q_indices=None, k_indices=None):
        if q_indices is None:
            qT = att.shape[-2]
            q_indices = torch.arange(start_query_index, qT + start_query_index, device=att.device)
        if k_indices is None:
            kT = att.shape[-1]
            k_indices = torch.arange(start_key_index, kT + start_key_index, device=att.device)
        return self._adapt_attention_before_softmax_for_indices(att, q_indices, k_indices)

    def _adapt_attention_before_softmax_for_indices(self, att, query_indices, key_indices):
        return att
    

class PositionalEncoder(nn.Module):

    closure_model = PositionalEncoderClosure

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x, self.closure_model(self)
