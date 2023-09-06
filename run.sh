# Symato tknz on vlc (tiny model)
#python3 train.py --load_model "" --wandb "" --proj_dir "out" \
#--data_file "../data/vlc.xyz" --data_type "symato" \
#--ctx_len 512 --epoch_steps 2000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
#--micro_bsz 24 --n_layer 4 --n_embd 320 --pre_ffn 0 --head_qk 0 \
#--lr_init 8e-4 --lr_final 1e-6 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
#--accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false
#   | Name   | Type       | Params
# --------------------------------------
# 0 | emb    | Embedding  | 942 K
# 1 | blocks | ModuleList | 5.3 M
# 2 | ln_out | LayerNorm  | 640
# 3 | head   | Linear     | 942 K
# --------------------------------------
# 7.2 M     Trainable params


# Symato tknz on vlc
# python train.py --data_order=reversed --load_model "" --wandb "" --proj_dir "out_large" \
# --data_file "/workspace/data/train.txt" --data_type "txt" \
# --ctx_len 640 --epoch_steps 2000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
# --micro_bsz 12 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
# --lr_init 8e-4 --lr_final 1e-6 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
# --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false
# --------------------------------------
#   | Name   | Type       | Params
# --------------------------------------
# 0 | emb    | Embedding  | 1.4 M
# 1 | blocks | ModuleList | 20.5 M
# 2 | ln_out | LayerNorm  | 1.0 K
# 3 | head   | Linear     | 1.4 M
# --------------------------------------
# 23.4 M    Trainable params
# 0         Non-trainable params
# 23.4 M    Total params


# PhoBert tknz on vlc
# cd data; wget https://huggingface.co/vinai/phobert-base/raw/main/vocab.txt; cd ..
# cd data; wget https://huggingface.co/vinai/phobert-base/raw/main/tokenizer.json; cd ..
# cd data; wget https://huggingface.co/vinai/phobert-base/raw/main/bpe.codes; cd ..
 python3 train.py --load_model "./out/rwkv-45.pth" --wandb "gptneo" --proj_dir "out" \
 --data_file "/workspace/data/train.txt" --data_type "utf-8" \
 --ctx_len 1024 --epoch_steps 2000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
 --micro_bsz 4 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
 --lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
 --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0
# --------------------------------------
#   | Name   | Type       | Params
# --------------------------------------
# 0 | emb    | Embedding  | 20.6 M
# 1 | blocks | ModuleList | 6.7 M
# 2 | ln_out | LayerNorm  | 640
# 3 | head   | Linear     | 20.6 M
# --------------------------------------
# 47.8 M    Trainable params
# 0         Non-trainable params
# 47.8 M    Total params
