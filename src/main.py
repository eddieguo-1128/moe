import evaluate
import numpy as np
from tqdm import tqdm 

import torch 

from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer 

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from transformers import HfArgumentParser

from arguments import MOEArguments as ModelArguments, \
    WMTArguments as DataArguments, \
    MOETrainingArguments as TrainingArguments

from utils import compute_metrics_factory


bleu = evaluate.load("bleu")

lang_map = {
    "cs": "Czech", 
    "de": "German", 
    "et": "Estonian", 
    "fi": "Finnish",
    "ru": "Russian", 
    "tr": "Turkish", 
    "zh": "Chinese", 
    "en": "English"
}


def load_and_tokenize_langpair(dataset_name, lang_config, tokenizer, max_length):
    # load splits
    train_dataset = load_dataset(dataset_name, lang_config, split="train") 
    valid_dataset = load_dataset(dataset_name, lang_config, split="validation")

    source_lang, target_lang = lang_config.strip().split("-")

    def preprocess_function(examples):
        prefix = f"translate {lang_map[source_lang]} to {lang_map[target_lang]}: "
        inputs = [prefix + example[source_lang] for example in examples['translation']]
        targets = [example[target_lang] for example in examples['translation']]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        return model_inputs

    # Map preprocessing
    tokenized_train = train_dataset.map(preprocess_function, batched=True).remove_columns(["translation"])
    tokenized_valid = valid_dataset.map(preprocess_function, batched=True).remove_columns(["translation"])

    return tokenized_train, tokenized_valid




def test_preprocess_function(tokenizer, examples, source_lang, target_lang, max_length):
    prefix = f"translate {lang_map[source_lang]} to {lang_map[target_lang]}: "
    inputs = [prefix + example for example in examples["translation"][source_lang]]
    targets = [example for example in examples["translation"][target_lang]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    model_inputs["labels"] = tokenizer(targets, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
    return model_inputs, inputs, targets




def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # init model 
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path) 
    model = SwitchTransformersForConditionalGeneration.from_pretrained(model_args.model_name_or_path)


    # init datasets from each language split
    train_sets = []
    valid_sets = []
    test_sets = []

    for lang in data_args.lang_config:
        train_data, valid_data = load_and_tokenize_langpair(
            dataset_name=data_args.dataset_name, 
            lang_config=lang, 
            tokenizer=tokenizer, 
            max_length=data_args.max_length
        )
        train_sets.append(train_data)
        valid_sets.append(valid_data)

    # combine all langs 
    combined_train_dataset = concatenate_datasets(train_sets)
    combined_valid_dataset = concatenate_datasets(valid_sets)

    # init dataloaders 
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # custom bleu score 
    compute_metrics = compute_metrics_factory(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_valid_dataset,
        eval_dataset=combined_valid_dataset.select(range(100)),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


    model.eval()
    model.to("cuda")

    for lang_config in data_args.lang_config: 

        all_preds = []
        all_refs = []
        
        source_lang, target_lang = lang_config.strip().split("-")
        test_dataset = load_dataset(data_args.dataset_name, lang_config, split="test")
        dataloader = DataLoader(test_dataset, batch_size=32)  

        for batch in tqdm(dataloader, desc=f"Inferencing {source_lang}->{target_lang}"):
            model_inputs, raw_inputs, targets = test_preprocess_function(tokenizer, batch, source_lang, target_lang, max_length=data_args.max_length)

            input_ids = model_inputs["input_ids"].to(model.device)
            attention_mask = model_inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_targets = targets  # already raw strings

            all_preds.extend(decoded_preds)
            all_refs.extend(decoded_targets)

        if sum([len(pred) for pred in all_preds]) == 0: 
            bleu_score = 0.0
        else: 
            bleu_score = bleu.compute(predictions=all_preds, references=all_refs)["bleu"]
        print(f"Average Test BLEU for {source_lang}->{target_lang}: {bleu_score}")


    


if __name__ == "__main__": 
    main()