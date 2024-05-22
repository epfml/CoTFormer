#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from data.utils import get_dataset, prepare_dataset

import torch
import models
import json
import os
import eval
import argparse

class Args:
    pass
def load_untrained_model(exp_path):
    with open(os.path.join(exp_path, "summary.json")) as f:
        config = Args()
        config.__dict__ = json.load(f)['args']
        
    if config.model == "but_halting_freeze_input_on_stop":
        config.model = "but_full_depth"
    if config.model == "adaptive_cotformer_halting":
        config.model = "cotformer_full_depth"
    if config.model == "but_mod_efficient_sigmoid_lnmid_depthemb_random_factor":
        config.model = "but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute"
    import models
    if not hasattr(config, "disable_ln_mid"):
        config.disable_ln_mid = False
    model = models.make_model_from_args(config)
    model.cuda()
    model.eval()
    return model

from ptflops import get_model_complexity_info
import functools

def _recreate_bias(module, seqlen):
    if "Attention" in module.__class__.__name__:
        module.bias = torch.tril(module.c_attn.weight.new_ones(seqlen, seqlen)).view(1, 1, seqlen, seqlen)
        if hasattr(module, "flash"):
            module.flash = False

@torch.no_grad()
def get_macs_for_seqlens(model, seq_lens):
    model_macs = []
    
    for seq_len in seq_lens:
        model.config.sequence_length = seq_len
        model.apply(functools.partial(_recreate_bias, seqlen=seq_len))
        macs, _ = get_model_complexity_info(
            model, 
            (seq_len,), 
            backend='aten',
            as_strings=False, 
            print_per_layer_stat=False, 
            input_constructor = lambda input_res: torch.ones((1, *input_res),
                                                             dtype=torch.long,
                                                             device=next(model.parameters()).device)
        )
        model_macs.append(macs)
    return model_macs





def run(checkpoint_dir):
    all_router_weights = np.load(f"{checkpoint_dir}/router_weights.npy", allow_pickle=True)
    results = {}
    model = load_untrained_model(checkpoint_dir)
    for threshold in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]:
        coefs = []
        for i in range(1, len(all_router_weights) ):
            coef = (all_router_weights[i] >= 1 - threshold).sum() / len(all_router_weights[i])
            coefs.append(np.ceil(coef * 256) / 256)
        args = [
            "--checkpoint", os.path.join(checkpoint_dir, "ckpt.pt"), "--distributed_backend", "None",
        ] + ["--eval_length_factor"] + ["{:.02f}".format(x) for x in coefs]
        
        model.config.eval_length_factor = [float("{:.02f}".format(x)) for x in coefs]
        model_macs = get_macs_for_seqlens(model, [256])[0]
        assert model_macs is not None
        stats = eval.main(eval.get_args(args))
        
        
        
        print(threshold, "@", " ".join("{:.02f}".format(x) for x in coefs), "@", stats, "@", model_macs)
        results[threshold] = (["{:.02f}".format(x) for x in coefs], stats, model_macs)
        print()
    with open(f"{checkpoint_dir}/eval_per_threshold.npy", "wb") as f:
        np.save(f, results)


    results = {}
    model = load_untrained_model(checkpoint_dir)
    for i in range(4):
        coefs = [1 if j <= i else 0 for j in range(4)]
        args = [
            "--checkpoint", os.path.join(checkpoint_dir, "ckpt.pt"), "--distributed_backend", "None",
        ] + ["--eval_length_factor"] + ["{:.02f}".format(x) for x in coefs]

        model.config.eval_length_factor = [float("{:.02f}".format(x)) for x in coefs]
        model_macs = get_macs_for_seqlens(model, [256])[0]
        assert model_macs is not None
        stats = eval.main(eval.get_args(args))
        
        
        #results[i] = (*results[i][:2], model_macs)
        print(i, "@", " ".join("{:.02f}".format(x) for x in coefs), "@", stats, "@", model_macs)
        results[i] = (["{:.02f}".format(x) for x in coefs], stats, model_macs)
        print()
    with open(f"{checkpoint_dir}/eval_per_layer.npy", "wb") as f:
        np.save(f, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=str, required=True)
    
    args, rem_args = parser.parse_known_args()
    run(args.checkpoint)
