
import time
import torch
from megatron import print_rank_0

try:
    from pynvml import *
    nvml_available = True
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
except ImportError:
    nvml_available = False

class BenchmarkLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.tokens_processed = 0
        torch.cuda.reset_peak_memory_stats()

    def start(self):
        self.start_time = time.time()

    def end(self, batch_size, seq_length):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        tokens = batch_size * seq_length
        self.tokens_processed += tokens
        throughput = tokens / elapsed

        max_allocated = torch.cuda.max_memory_allocated() / 1e6  # MB
        log_str = f"[BENCH] Throughput: {throughput:.2f} tokens/sec | Max Memory: {max_allocated:.2f} MB"

        if nvml_available:
            util = nvmlDeviceGetUtilizationRates(handle)
            log_str += f" | GPU Utilization: {util.gpu}%"

        print_rank_0(log_str)

    def log_expert_usage(self, gate_scores):
        expert_usage = gate_scores.sum(dim=0).tolist()
        usage_str = ", ".join(f"{u:.1f}" for u in expert_usage)
        print_rank_0(f"[BENCH] Expert usage: [{usage_str}]")
