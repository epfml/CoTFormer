import argparse
import numpy as np
from tqdm import tqdm
from data.utils import get_dataset, prepare_dataset
from contextlib import nullcontext

import torch
import models
import json
import os

def iceildiv(x, y):
    return (x + y - 1) // y


def get_as_batch(data, seq_length, batch_size, device='cpu', sample_size=None):
    all_ix = list(range(0, len(data), seq_length))
    assert all_ix[-1] + seq_length + 1 > len(data)
    all_ix.pop()
    if sample_size is not None:
        all_ix = np.random.choice(all_ix, size=sample_size // seq_length, replace=False).tolist()
    
    idx = 0
    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def forward(self, idx, targets):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
    
    
    # forward the GPT model itself
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
    
    active_indices = (index_shift + torch.arange(T, device=x.device)).unsqueeze(0).repeat(B, 1).view(B, T)
    
    router_weights = None
    
    final_mask = []
    all_outputs = []
    all_indices = []

    active_mask = x.new_ones((B, T)) == 1.
    sum_active = 0
    all_router_weights = []
    for rep_idx in range(1, self.n_repeat+1):
        x_in = x
        if self.depth_emb is not None:
            x = self.depth_emb(x, indices=torch.full_like(active_indices, self.n_repeat - rep_idx))
        for block in self.transformer.h_mid:
            x = block(x, pos_emb_closure, cache_context, start_index=None, indices=active_indices)
            x = torch.where(active_mask.unsqueeze(-1), x, x_in)
        x = self.transformer.ln_mid(x)
        if router_weights is not None:
            x = x_in * (1 - router_weights) + x * router_weights
            all_router_weights.append(torch.where(active_mask.unsqueeze(-1), router_weights, 0.))

        sum_active += x.shape[1]
        if rep_idx < self.n_repeat:
            is_final, router_weights = self.transformer.mod[rep_idx - 1](x, active_mask, capacity_factor=1.0)
            active_mask = active_mask & (~is_final)
        
    return all_router_weights

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True
    if os.path.isdir(args.checkpoint):
        args.checkpoint = args.checkpoint + '/'
    checkpoint_dir, checkpoint_filename = os.path.split(args.checkpoint)
    if not checkpoint_filename:
        checkpoint_filename = 'ckpt.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    class Args:
        pass
    with open(os.path.join(checkpoint_dir, "summary.json")) as f:
        config = Args()
        config.__dict__ = json.load(f)['args']
    model = models.make_model_from_args(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(
        {x: y 
        for x, y in checkpoint['model'].items() 
        if "attn.bias" not in x and "wpe" not in x}, strict=False)
    data = get_dataset(config)
    config.device = "cuda:0"
    device_type = 'cuda' if 'cuda' in str(config.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
            device_type=device_type, dtype=config.dtype)  # extra_args.dtype)
    model.cuda()
    model.eval()

    all_router_weights = [[] for _ in range(model.n_repeat)]
    for idx, (x, y) in tqdm(enumerate(get_as_batch(
            data['train'], 
            config.sequence_length, 
            config.batch_size, 
            device=config.device,
            sample_size=len(data['val']),
            )),
            total=iceildiv(
                iceildiv(len(data['val']), config.sequence_length), 
                config.batch_size
            )
        ):
            with torch.no_grad():
                with type_ctx:
                    router_weights = forward(model, x, y)
            for i in range(2, len(router_weights)):
                for j in range(len(router_weights[i])):
                    router_weights[i][j] = torch.minimum(router_weights[i][j], router_weights[i - 1][j])
            for x, y in zip(all_router_weights, router_weights):
                if y is not None:
                    x += y.detach().view(-1).tolist()
    all_router_weights = [np.array(x) if x is not None else x for x in all_router_weights]
    output_path = os.path.join(checkpoint_dir, "router_weights.npy")
    with open(output_path, "wb") as f:
        np.save(f, all_router_weights)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=str, required=True)
    
    args, rem_args = parser.parse_known_args()
    main(args)
