#!/usr/bin/env python3

import torch
import argparse

from megatron import get_args, print_rank_0
from megatron.training import pretrain
from megatron.model import GPTModel
from megatron.data.gpt_dataset import build_train_valid_test_datasets

from benchmark_logger import BenchmarkLogger
bench = BenchmarkLogger()

def build_datasets(args):
    args = get_args()
    
   # print_rank_0(f"[DEBUG] args.data_path: {args.data_path}")
   # print_rank_0(f"[DEBUG] type: {type(args.data_path)}")
    data_prefix = []
    for path in args.data_path:
        # Add weight and path as separate items in the list
        data_prefix.extend([1.0, path])
   # print_rank_0(f"[DEBUG] FINAL data_prefix: {data_prefix}")
   # print_rank_0(f"[DEBUG] FINAL type: {type(data_prefix)}")
    # Build datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=data_prefix,
        data_impl=args.data_impl,
        splits_string=args.split,
        seq_length=args.seq_length,
        train_valid_test_num_samples=[10000000, 50000, 5000],
        seed=args.seed,
        skip_warmup=True
    )

    # Print number of samples in each split
    from megatron import print_rank_0
    print_rank_0(f"Dataset sample sizes:\n"
                 f"  Train: {len(train_ds)}\n"
                 f"  Valid: {len(valid_ds)}\n"
                 f"  Test:  {len(test_ds)}")

    return train_ds, valid_ds, test_ds

# FastMoE MLP only
from fmoe import FMoETransformerMLP

# Create an adapter class to match Megatron's interface
# Create an adapter class to match Megatron's interface
class MegatronFMoEAdapter(torch.nn.Module):
    def __init__(self, num_experts, d_model, d_hidden, top_k):
        super().__init__()
        self.moe_mlp = FMoETransformerMLP(
            num_expert=num_experts,
            d_model=d_model,
            d_hidden=d_hidden,
            world_size=1,  # Can be set from dist group size if using multi-node
            top_k=top_k,
            activation=torch.nn.functional.gelu
        )
        # Create a proper bias vector (zero-initialized)
        self.bias = torch.nn.Parameter(torch.zeros(d_model))
        
    def forward(self, hidden_states):
        # FMoE returns only the output tensor
        output = self.moe_mlp(hidden_states)
        # Return output and the bias tensor
        return output, self.bias

def forward_step_func(data_iterator, model, input_tensor=None):
    args = get_args()
    bench.start()
    # Get the data
    data = next(data_iterator)
    tokens = data['text'].cuda()
    
    batch_size, seq_length = tokens.size()
    position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
   # print(f"[DEBUG] tokens shape: {tokens.shape}, max token ID: {tokens.max().item()}")
   # print(f"[DEBUG] position_ids shape: {position_ids.shape}, max position ID: {position_ids.max().item()}")

    # Unwrap model from DDP and FP16Module
    real_model = model
    if hasattr(real_model, "module"):
        real_model = real_model.module
    if hasattr(real_model, "module"):
        real_model = real_model.module

    # Safe clamping
    vocab_size = real_model.language_model.embedding.word_embeddings.num_embeddings
    max_position = real_model.language_model.embedding.position_embeddings.num_embeddings

    if tokens.max() >= vocab_size:
       # print(f"[WARNING] Clamping tokens. vocab_size={vocab_size}")
        tokens = tokens.clamp(max=vocab_size - 1)

    if position_ids.max() >= max_position:
       # print(f"[WARNING] Clamping position_ids. max_position={max_position}")
        position_ids = position_ids.clamp(max=max_position - 1)
    
    # Create attention mask with correct type
    attention_mask = torch.ones(
        batch_size, 1, seq_length, seq_length,
        device=tokens.device,
        dtype=torch.bool  # ← Important!
    )
    
    output = model(tokens, position_ids, attention_mask)

    labels = tokens.clone()
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    bench.end(batch_size, seq_length)
    
    return loss, {'loss': loss}




def add_moe_args(parser):
    """Add MOE-specific arguments to the parser."""
    group = parser.add_argument_group(title='MoE Arguments') # Optional: group args for cleaner help message
    group.add_argument('--num-experts', type=int, default=1, # Or None if required
                         help='Number of experts per MoE layer.')
    group.add_argument('--top-k', type=int, default=1, # Or None if required
                         help='Number of experts to route to per token.')
    # Add any other MoE specific args here if needed
    return parser

def model_provider():
    """Build the model."""
    print_rank_0('Building MOE GPT model...')
    args = get_args()

    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True
    )

    print_rank_0(f'Replacing MLP layers with MOE layers (num_experts={args.num_experts}, top_k={args.top_k})')

    # Replace standard MLP with FMoE MLP
    for i, layer in enumerate(model.language_model.transformer.layers):
        orig_mlp = layer.mlp
        d_model = orig_mlp.dense_h_to_4h.weight.size(1)
        d_hidden = orig_mlp.dense_h_to_4h.weight.size(0)

        moe_mlp = MegatronFMoEAdapter(
            num_experts=args.num_experts,
            d_model=d_model,
            d_hidden=d_hidden,
            top_k=args.top_k
        )

        layer.mlp = moe_mlp
        print_rank_0(f'  Replaced MLP in layer {i} with MOE MLP')

    return model

def main():
    """Main training program."""
    pretrain(build_datasets, model_provider, forward_step_func, extra_args_provider=add_moe_args)


if __name__ == "__main__":
    main()

