#!/bin/bash

python get_activations.py \
    Qwen1.5-14B-Chat \
    DRC \
    --model_dir "./models/Qwen1.5-14B-Chat"

python edit_weight.py \
    --model_name Qwen1.5-14B-Chat \
    --dataset_name DRC \
    --activation_path "features/Qwen1.5-14B-Chat_DRC_head_wise.npy" \
    --label_path "features/Qwen1.5-14B-Chat_DRC_labels.npy" \
    --model_dir "./models/Qwen1.5-14B-Chat" \
    --num_heads 64 \
    --alpha 3

python generate.py "edited_model/Qwen1.5-14B-Chat_dataset_DRC_seed_42_top_64_heads_alpha_3.0"
