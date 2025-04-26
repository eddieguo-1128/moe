#!/bin/bash

# Set up directories
export DATA_PATH=$PWD/data_wikitext103
export CHECKPOINT_PATH=$PWD/wikitext103_checkpoints
export TENSORBOARD_PATH=$PWD/wikitext103_tensorboard
export MEGATRON_PATH=$PWD/Megatron-LM

mkdir -p $DATA_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $TENSORBOARD_PATH

echo "Downloading WikiText-103 dataset using Hugging Face Datasets library..."

python - <<EOF
from datasets import load_dataset
import json
import os

dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')


def write_jsonl(split, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset[split]:
            text = item['text'].strip()
            if text:  # Skip empty lines
                json.dump({'text': text}, f)
                f.write('\n')

write_jsonl('train', os.path.join('$DATA_PATH', 'wiki.train.jsonl'))
write_jsonl('validation', os.path.join('$DATA_PATH', 'wiki.valid.jsonl'))
write_jsonl('test', os.path.join('$DATA_PATH', 'wiki.test.jsonl'))

print("WikiText-103 dataset downloaded and saved as JSONL.")
EOF

echo "Preprocessing WikiText-103 dataset for Megatron-LM..."

# Preprocess training set
python $MEGATRON_PATH/tools/preprocess_data.py \
    --input $DATA_PATH/wiki.train.jsonl \
    --output-prefix $DATA_PATH/wikitext103_train \
    --vocab-file $MEGATRON_PATH/megatron/tokenizer/gpt2-vocab.json \
    --merge-file $MEGATRON_PATH/megatron/tokenizer/gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --json-keys text \
    --workers 4

# Preprocess validation set
python $MEGATRON_PATH/tools/preprocess_data.py \
    --input $DATA_PATH/wiki.valid.jsonl \
    --output-prefix $DATA_PATH/wikitext103_valid \
    --vocab-file $MEGATRON_PATH/megatron/tokenizer/gpt2-vocab.json \
    --merge-file $MEGATRON_PATH/megatron/tokenizer/gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --json-keys text \
    --workers 4

# Preprocess test set
python $MEGATRON_PATH/tools/preprocess_data.py \
    --input $DATA_PATH/wiki.test.jsonl \
    --output-prefix $DATA_PATH/wikitext103_test \
    --vocab-file $MEGATRON_PATH/megatron/tokenizer/gpt2-vocab.json \
    --merge-file $MEGATRON_PATH/megatron/tokenizer/gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --json-keys text \
    --workers 4

echo "WikiText-103 dataset downloaded and preprocessed successfully."
