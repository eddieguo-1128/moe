#!/bin/bash

#SBATCH --job-name=run
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4=2

#SBATCH --mem=32G

#SBATCH --gres=gpu:1

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"


exp_name="trial"

learning_rate=1e-4
n_epochs=1
model_name="google/switch-base-8"
num_experts=8
top_k=1


output_dir="/data/user_data/jingyuah/moe/${exp_name}" 


python src/main.py \
    --model_name_or_path $model_name \
    --dataset_name "wmt/wmt18" \
    --num_experts $num_experts \
    --top_k $top_k  \
    --lang_config "tr-en" \
    --num_train_epochs $n_epochs  \
    --learning_rate $learning_rate \
    --warmup_ratio 0.1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --eval_strategy "steps"  \
    --logging_strategy "steps"  \
    --save_strategy "steps"  \
    --metric_for_best_model "eval_loss" \
    --logging_steps 10 \
    --save_steps 100  \
    --eval_steps 100  \
    --output_dir $output_dir \
    --save_total_limit 2 \
    --seed 42  