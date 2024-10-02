# DRESS-LLM
Code and Benchmark Dataset for paper DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Workflow](#workflow)


## Installation
Run the following commands to set things up.
```
git clone XXXX (This Github Link)
cd DRESS-LLM
conda env create -f environment.yml
conda activate DRESSllm
```

## Dataset

`dataset/Train_Shakespeare.json` and `dataset/Train_DRC.json` are the two training datasets we created for the language style transfer task, namely the Shakespearean style and the dialogue style of characters in Dream of the Red Chamber. For each piece of data,  `question`, `correct_answers`, and `incorrect_answers` are respectively input, target style output, and ordinary style output (i.e., generic featureless output), and the two outputs are almost semantically equivalent. 

The testing sets are `dataset/Valid_Shakespeare.json` and `dataset/Valid_DRC.json` respectively. We only use the `question` for testing.

For more information about the dataset, please refer to the paper.

## Workflow

(1) Get activations by running `python get_activations.py Qwen1.5-14B-Chat DRC --model_dir "/PretainedModels/Qwen1.5-14B-Chat"`. You need to fill in the model name (Qwen1.5-14B-Chat in this example), training set name(DRC or Shakespeare), and model path("/PretainedModels/Qwen1.5-14B-Chat" in this example) in this instruction. The steering vectors extracted from the training set will be saved in the `features` folder.

(2) Run `python edit_weight.py --model_name Qwen1.5-14B-Chat --dataset_name DRC --activation_path "features/Qwen1.5-14B-Chat_DRC_head_wise.npy" --label_path "features/Qwen1.5-14B-Chat_DRC_labels.npy" --model_dir "/PretainedModels/Qwen1.5-14B-Chat" --num_heads 64 --alpha 3` to edit the llm and save it. You also need to fill in the model name and path, as well as the dataset name. For the Shakespeare dataset, simply replace all 'DRC's in the example with 'Shakespeare'. Parameters num_heads and alpha specify the number of edited heads and the steering intensity, respectively. The edited model is stored in the `edited_model` folder and can be directly used for inference.

(3) Run `python generate.py "edited_model/Qwen1.5-14B-Chat_dataset_DRC_seed_42_top_64_heads_alpha_3.0"` to perform inference on the test set and generate answers to all questions. Only the model path in (2) needs to be provided here. The reasoning adopts the [DRESSing UP LLM] strategy, adaptively adjusting the steering intensity in the style subspace to achieve higher generation quality.
Results will be saved in `result.json`.
