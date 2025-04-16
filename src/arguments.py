import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

from typing import List


@dataclass 
class DataArguments(): 
    dataset_name: str = field(metadata={"help": "huggingface dataset name"})
    max_length: int = field(default=128, metadata={"help": "maximum length of the input"})
    # train_batch_size: int = field(default=128, metadata={"help": "train batch size"})
    # eval_batch_size: int = field(default=128, metadata={"help": "eval batch size"})
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass 
class WMTArguments(DataArguments): 
    dataset_name: str = field(
        default='wmt/wmt18', metadata={"help": "huggingface dataset name"}
    )
    lang_config: list[str] = field(
        default_factory=lambda: ["cs-en", "de-en", "et-en", "fi-en", "ru-en", "tr-en", "zh-en"], metadata={"help": "language splits to load"}
    )

@dataclass
class MOEArguments(ModelArguments):
    num_experts: int = field(default=16, metadata={"help": "Number of experts in the model"})
    top_k: int = field(default=1, metadata={"help": "Top-k experts to use in top-k routing"})


@dataclass
class MOETrainingArguments(TrainingArguments):
    pass 