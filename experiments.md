# Standard Transformer
## 12 Layer
```bash
python ./main.py \
    --config_format base \
    --distributed_backend nccl \
    --model base \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 256 \
    --acc_steps 2 \
    --data_in_ram \
    --dropout 0.0 \
    --compile \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --seed 0 \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --warmup_percent 0.2
```
## 24 Layer
```bash
python ./main.py \
    --config_format base \
    --distributed_backend nccl \
    --model base \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 64 \
    --sequence_length 256 \
    --acc_steps 2 \
    --data_in_ram \
    --dropout 0.0 \
    --compile \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --seed 0 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --warmup_percent 0.2
```
## 48 Layer
```bash
python ./main.py \
    --config_format base \
    --distributed_backend nccl \
    --model base \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 48 \
    --batch_size 64 \
    --sequence_length 256 \
    --acc_steps 2 \
    --data_in_ram \
    --dropout 0.0 \
    --compile \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --seed 0 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --warmup_percent 0.2
```
# CoTFormer - 12 Layers
## 12x2
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 2 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 2
```
## 12x3
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 3 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 3
```
## 12x5
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 5
```
## 12x15
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 16 \
    --sequence_length 256 \
    --acc_steps 8 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 15 \
    --warmup_percent 0.2 \
    --seed 0 \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --distributed_backend nccl \
    --depth_embedding linear_learned
```
# Block Universal - 12 Layers
## 12x2
```bash
python ./main.py \
    --config_format base \
    --model but_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 2 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --compile \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 2
```
## 12x3
```bash
python ./main.py \
    --config_format base \
    --model but_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 3 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --compile \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 3
```
## 12x5
```bash
python ./main.py \
    --config_format base \
    --model but_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --compile \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 5
```
## 12x6
```bash
python ./main.py \
    --config_format base \
    --model but_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 6 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --compile \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 6
```
## 12x15
```bash
python ./main.py \
    --config_format base \
    --model but_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 16 \
    --sequence_length 256 \
    --acc_steps 8 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 15 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --compile \
    --save_checkpoint_freq 10000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 15
```
# CoTFormer - 24 Layers
## 24x2
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 2 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 2
```
## 24x3
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 3 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 3
```
## 24x5
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 0 \
    --n_layer_end 0 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 5
```
# Ablation CoTFormer
## Reservered Layers
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 5
```
## LN-CoTFormer
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth_lnmid_depthemb \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 5
```
## Depth Embedding
```bash
python ./main.py \
    --config_format base \
    --model cotformer_full_depth_lnmid_depthemb \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method uniform_random_range \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --min_repeat 5 \
    --depth_embedding linear_learned
```
# Adaptive LN-CoTFormer
## Without Depth Embedding
```bash
python ./main.py \
    --config_format base \
    --model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --distributed_backend nccl
```
## With Depth Embedding
```bash
python ./main.py \
    --config_format base \
    --model adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 40000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --distributed_backend nccl \
    --depth_embedding linear_learned
```
# Other Adaptive Methods Ablation
## PonderNet
```bash
python ./main.py \
    --config_format base \
    --distributed_backend nccl \
    --model pondernet \
    --ponder_lambda_p 0.4 \
    --ponder_kl_div_loss_weight 0.2 \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 10000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end
```
## Stick Breaking
```bash
python ./main.py \
    --config_format base \
    --model but_halting_freeze_input_on_stop \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 10000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --distributed_backend nccl \
    --depth_random_method sb_act \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --compile \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end
```
## Mixture of Repeats
```bash
python ./main.py \
    --config_format base \
    --model but_mod_efficient_sigmoid_lnmid_depthemb_random_factor \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 24 \
    --batch_size 32 \
    --sequence_length 256 \
    --acc_steps 4 \
    --dropout 0.0 \
    --iterations 10000 \
    --dataset owt2 \
    --lr 1e-3 \
    --weight_decay 0.1 \
    --eval_freq 100 \
    --n_repeat 5 \
    --warmup_percent 0.2 \
    --seed 0 \
    --n_layer_begin 2 \
    --n_layer_end 1 \
    --save_checkpoint_freq 1000 \
    --remove_intermediary_checkpoints_at_end \
    --distributed_backend nccl \
    --disable_ln_mid
```
