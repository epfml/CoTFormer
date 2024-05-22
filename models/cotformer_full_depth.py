"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import positional_encoders, caches

from .utils import LayerNorm


class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_val, dim):
        
        if last_slice is None:
            prev_length = 0
        else:
            prev_length = last_slice.shape[dim]
        new_length = prev_length + x_val.shape[dim]

        prefix_slice = [slice(None)] * dim 
        full_tensor[prefix_slice + [slice(prev_length, new_length)]] = x_val
        ctx.prev_length = prev_length
        ctx.new_length = new_length
        ctx.dim = dim
        ret = torch.Tensor().to(full_tensor)
        ret.set_(full_tensor[prefix_slice +[slice(None,new_length)]])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        prefix_slice = [slice(None)] * ctx.dim 
        if ctx.prev_length == 0:
            return None, None, grad_out[prefix_slice + [slice(None, ctx.new_length)]], None
        else:
            return None, grad_out[prefix_slice + [slice(None, ctx.prev_length)]], grad_out[prefix_slice + [slice(ctx.prev_length, ctx.new_length)]], None


def apply_inplace_set(x_acc, x_val, dim):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_val, dim)
    return full_tensor, new_slice


class CausalSelfAttention(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.cache_storage = lm_cache.get_storage_for_layer(self)
        self.config = config
        self.allow_cache_during_training = getattr(config, "allow_cache_during_training", False)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            assert config.attention_window_length is None
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
        if config.attention_window_length is not None:
            bias = torch.triu(bias, diagonal=-config.attention_window_length)
        self.register_buffer("bias", bias.view(1, 1, config.sequence_length, config.sequence_length))

        self.drop_cache()

    def init_cache(self, expected_total_length):
        self._lazy_init_cache_length = expected_total_length

    def drop_cache(self):
        self.all_keys = None
        self.all_values = None
        self.all_indices = None
        self._lazy_init_cache_length = None
        

    def forward(self, x, pos_emb_closure, cache_context, start_index, indices):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = self.n_embd
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        pos_size = k.shape[-1] // 2

        q = pos_emb_closure.adapt_queries(q, start_index=start_index, indices=indices)
        if cache_context is not None and self.cache_storage is not None:
            att_prefix, cache_values_dict = \
                self.cache_storage.retrieve_for_query(q, cache_context, pos_emb_closure, start_index)
            if self.training and att_prefix is not None and not self.allow_cache_during_training:
                raise ValueError("Cache is not allowed during training")
        else:
            att_prefix = None
        k_before_pos = k
        k = pos_emb_closure.adapt_keys(k, start_index=start_index, indices=indices)

        if self._lazy_init_cache_length is not None:
            # assert indices is not None
            self.all_keys = (
                k.new_empty((B, self.n_head, self._lazy_init_cache_length, C // self.n_head)),
                None
            )
            self.all_values = (
                v.new_empty((B, self.n_head, self._lazy_init_cache_length, C // self.n_head)),
                None
            )
            self._lazy_init_cache_length = None
        
        if self.all_keys is not None:
            # assert indices is not None
            self.all_keys = apply_inplace_set(self.all_keys, k, dim=2)
            self.all_values = apply_inplace_set(self.all_values, v, dim=2)
            k = self.all_keys[1]
            v = self.all_values[1]
            attn_mask = self.bias[:,:,:T,:T].unsqueeze(3).repeat(
                1, 1, 1, k.shape[2] // T, 1
            ).unsqueeze(0).view(1, 1, q.shape[2], k.shape[2]) == 1
            is_causal = False
        else:
            attn_mask = None
            is_causal = True
        
        if self.flash:
            if att_prefix is not None:
                raise NotImplementedError
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = pos_emb_closure.adapt_attention_before_softmax(att, start_query_index=start_index, start_key_index=start_index)
            if attn_mask is None:
                attn_mask = self.bias[:,:,:T,:T] == 1
            att = att.masked_fill(~attn_mask, float('-inf'))
            if att_prefix is not None:
                prefix_size = att_prefix.shape[-1]
                current_size = att.shape[-1]
                att = torch.cat((att_prefix, att), dim=-1)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            if att_prefix is not None:
                att_prefix, att = torch.split(att, (prefix_size, current_size), dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            if att_prefix is not None:
                cache_v = cache_values_dict['v']
                if cache_v.ndim == v.ndim:
                    y += att_prefix @ cache_v
                elif cache_v.ndim == v.ndim + 1:
                    y += (att_prefix.unsqueeze(3) @ cache_v).squeeze(3)
                else:
                    raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if cache_context is not None and self.cache_storage is not None:
            with torch.no_grad():
                self.cache_storage.store_in_cache(k_before_pos, {'v': v})
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, lm_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, pos_emb_closure, cache_context, start_index, indices=None):
        x = x + self.attn(self.ln_1(x), pos_emb_closure, cache_context, start_index, indices)
        x = x + self.mlp(self.ln_2(x))
        return x



class GPTBase(nn.Module):

    needs_iter = False

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.n_repeat = config.n_repeat

        self.lm_cache = caches.get_cache(config.lm_cache)(config)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = positional_encoders.get_encoder(config.positional_encoder)(config),
            drop = nn.Dropout(config.dropout),
            h_begin = nn.ModuleList( 
                [Block(config, self.lm_cache) for _ in range(config.n_layer_begin)]
            ),
            h_mid = nn.ModuleList(
                [Block(config, self.lm_cache) 
                for _ in range(config.n_layer_begin, config.n_layer - config.n_layer_end)],
            ),
            h_end = nn.ModuleList( 
                [Block(config, self.lm_cache) 
                for _ in range(config.n_layer - config.n_layer_end, config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        def _post_init_fn(module):
            if hasattr(module, "post_init"):
                module.post_init()
        self.apply(_post_init_fn)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wpe.parameters()) # TODO: Why do we need this?
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False, use_cache=False, iter=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        
        
        # forward the GPT model itself
        if use_cache:
            idx, index_shift, cache_context = self.lm_cache(idx)
        else:
            index_shift = 0
            cache_context = None
        if getattr(self.transformer.wpe, "needs_iter", False):
            idx, pos_emb_closure = self.transformer.wpe(idx, iter=iter) # position embeddings of shape (1, t, n_embd)
        else:
            idx, pos_emb_closure = self.transformer.wpe(idx) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(x)
        x = pos_emb_closure.adapt_model_input(x, start_index=index_shift)
       
        for block in self.transformer.h_begin:
            x = block(x, pos_emb_closure, cache_context, start_index=index_shift)
        
        B, T, D = x.shape

        total_expected_length = self.n_repeat * T
        for block in self.transformer.h_mid:
            block.attn.init_cache(total_expected_length)
        sum_active = 0
        for rep_idx in range(1, self.n_repeat+1):
            
            for block in self.transformer.h_mid:
                x = block(x, pos_emb_closure, cache_context, start_index=index_shift)

        for block in self.transformer.h_mid:
            block.attn.drop_cache()

        for block in self.transformer.h_end:
            x = block(x, pos_emb_closure, cache_context, start_index=index_shift)
        
        x = self.transformer.ln_f(x)

        if use_cache:
            x = self.lm_cache.get_final_logits(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            cross_entropy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = cross_entropy_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            cross_entropy_loss = None
        logits = logits if get_logits else None
        return {'logits': logits, 'loss': loss, 'cross_entropy_loss': cross_entropy_loss, 'average_depth': torch.as_tensor(self.n_repeat) * len(self.transformer.h_mid) + len(self.transformer.h_begin) + len(self.transformer.h_end)}

    def clear_state(self):
        self.lm_cache.clear_state()

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
