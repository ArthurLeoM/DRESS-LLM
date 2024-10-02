# python edit_weight.py --model_name Qwen1.5-14B-Chat --dataset_name DRC --activation_path "" --label_path "" --model_dir "/data/CharacterAI/PretainedModels/Qwen1.5-14B-Chat" --num_heads 64 --alpha 3
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import sys
sys.path.append('../')
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama
import qwen2


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen1.5-14B-Chat', help='model name')
    parser.add_argument("--dataset_name", type=str, default=None, help='dataset name')
    parser.add_argument("--activation_path", type=str, default=None, help='activation path')
    parser.add_argument("--label_path", type=str, default=None, help='label path')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    # 以上为必需参数
    parser.add_argument('--num_heads', type=int, default=96, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=5, help='alpha, intervention strength')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    # set seeds
    print("set seeds")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    print("create model")
    MODEL = args.model_dir
    tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(MODEL)
    model = qwen2.Qwen2ForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    print("load activations")
    head_wise_activations = np.load(f"{args.activation_path}")
    labels = np.load(f"{args.label_path}")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    print(head_wise_activations.shape)
    
    dataset_len = head_wise_activations.shape[0] // 2

    # tuning dataset: no labels used, just to get std of activations along the direction
    tuning_activations = np.load(f"{args.activation_path}")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    tuning_labels = np.load(f"{args.label_path}")

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    train_idxs = np.arange(dataset_len)

    # pick a val set using numpy
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    # get directions
    com_directions = None
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
    np.save(f"features/probes_{args.num_heads}_{args.alpha:.1f}.npy",probes)
    np.save(f"features/top_heads_{args.num_heads}_{args.alpha:.1f}.npy",top_heads)

    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

    activations_dict = {} # save
    for head_out_name, list_int_vec in tqdm(interventions.items()):
        layer_no = int(head_out_name.split('.')[2])
        displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
        activations_dict[layer_no] = {} # save
        for head_no, head_vec, std in list_int_vec:

            activations = tuning_activations[:,layer_no,head_no,:]
            correct_activations = activations[::2, :]
            incorrect_activations = activations[1::2, :]
            correct_activations = np.mean(correct_activations, axis=0)
            incorrect_activations = np.mean(incorrect_activations, axis=0)
            displacement[head_no] = args.alpha * (correct_activations - incorrect_activations)
            
            activations_dict[layer_no][head_no] = displacement[head_no] # save
      
        device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
        model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)
    with open(f"features/activations_{args.num_heads}_{args.alpha:.1f}.pkl", 'wb') as f:
        pickle.dump(activations_dict, f)

    print("save results")
    save_folder = f"edited_model/{args.model_name}_dataset_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha:.1f}"
    if os.path.exists(save_folder):
      shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    model.config.oproj_bias = True
    model.save_pretrained(save_folder, safe_serialization=False, max_shard_size="10GB")
    tokenizer.save_pretrained(save_folder)


if __name__ == "__main__":
    main()
