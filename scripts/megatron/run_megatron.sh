#!/bin/bash
#SBATCH --job-name=moe_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-16:1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --account=cis240126p
#SBATCH --partition=GPU-shared
#SBATCH --output=moe_test_%j.out
#SBATCH --error=moe_test_%j.err

echo "Activating environment..."
module load cuda/12.4.0
module load gcc

export CUDA_HOME=$(dirname $(dirname $(which nvcc))) export PATH=$CUDA_HOME/bin:$PATH export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Force use of newer GCC libstdc++
export LD_LIBRARY_PATH=/opt/packages/gcc/v13.3.1-p20240614/b2gpu/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/packages/gcc/v13.3.1-p20240614/b2gpu/lib64/libstdc++.so.6

echo "Setting up environment paths..."
export PYTHONPATH=$PWD/fastmoe:$PWD/Megatron-LM:$PYTHONPATH
export MEGATRON_PATH=$PWD/Megatron-LM
export DATA_PATH=$PWD/data_wikitext103
export CHECKPOINT_PATH=$PWD/checkpoints
export TENSORBOARD_PATH=$PWD/tensorboard
export MEGATRON_DISABLE_FUSED_KERNELS=1


mkdir -p $CHECKPOINT_PATH $TENSORBOARD_PATH

python src/megatron_moe.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 2 \
  --hidden-size 256 \
  --num-attention-heads 4 \
  --seq-length 128 \
  --max-position-embeddings 128 \
  --micro-batch-size 2 \
  --global-batch-size 4 \
  --train-iters 10000 \
  --eval-iters 200 \
  --log-interval 10 \
  --lr 1e-4 \
  --data-path $DATA_PATH/wikitext103_train_text_document \
  --vocab-file $MEGATRON_PATH/megatron/tokenizer/gpt2-vocab.json \
  --merge-file $MEGATRON_PATH/megatron/tokenizer/gpt2-merges.txt \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --tokenizer-type GPT2BPETokenizer \
  --fp16 \
  --checkpoint-activations \
  --save $CHECKPOINT_PATH \
  --save-interval 1000 \
  --tensorboard-dir $TENSORBOARD_PATH \
  --num-experts 4 \
  --top-k 2 \
  --DDP-impl local \
  --no-scaled-masked-softmax-fusion

echo "Run complete."
