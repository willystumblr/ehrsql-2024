import json
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
import os
import torch
import random
import argparse
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from utils.settings import PADDING_MAP, set_seed, wandb_setup, huggingface_login, LLMSampleCB, HF_W_TOKEN
from peft import LoraConfig, PeftModel # get_peft_model
# from trl.trainer import get_kbit_device_map, get_quantization_config
from utils.data_io import (
    BASE_DATA_DIR,
    BASE_CKPT_DIR,
)
from utils.args import add_default_args
from utils.prompt import create_eval_prompt_batch, create_prompt, create_sample_prompt
from utils.data_io import read_json as read_data
from utils.data_io import write_json as write_data
from utils.data_io import build_dataset
from accelerate import Accelerator
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
"""
python sft.py \
    --train_type=unanswerable \
    --base_model_name=google/gemma-2b-it \
    --project_name=ehrsql-2024-sft-text2sql \
    --model_name=willystumblr/ehrsql-2024-sft-unanswerable-gemma-2b-it \
    --train_epochs=3 \
    --train_batch_size=8 \
    --learning_rate=1e-3 \
    --logging_steps=10 \
    --lr_scheduler_type=cosine \
    --bf16=1 \
    --db_id=mimic_iv \
    --evaluation_strategy=no \
    --test_batch_size=1 \
    --save_strategy=epoch \
    --adapter_config_path=adapter_config/gemma.json \
    --phase=test
"""

if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args = parser.parse_args()
    # Determine device for training and set model save path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.output_dir = f'{BASE_CKPT_DIR}/{args.train_type}'

    # Set random seed for reproducibility
    set_seed(args)

    train_data, valid_data, test_data = build_dataset(args)
    


    ### WandB setting
    run = wandb_setup(args)
    huggingface_login()

    os.environ["WANDB_PROJECT"] =  args.project_name # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Configure CUDA settings
    # This code is originally written for Google Colab
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))

    if not torch.backends.cudnn.benchmark:
        torch.backends.cudnn.benchmark = True

    save_dir = f"{args.output_dir}/{wandb.run.name}"
    repo_id = f"{args.project_name}-{args.base_model_name.split('/')[-1]}"
    
    peft_parameters = LoraConfig(
        **read_data(args.adapter_config_path)
    )
    
    
    model_config = dict(
                id2label={0: "unanswerable", 1: "answerable"},
                num_labels=2,
                label2id={"unanswerable":0, "answerable":1},
                # problem_type="multi_label_classification"
            )
    save_dir = f"{args.output_dir}/{wandb.run.name}"
    
    model_name = args.model_name if args.model_name else args.base_model_name
    padding_side = PADDING_MAP[model_name] if model_name in PADDING_MAP.keys() else 'left'
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_config)
    model.add_adapter(peft_parameters)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=args.max_seq_length)
    
    
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    # Instantiate dataloaders.
    
    train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=['id', 'type', 'question', 'text'])
    valid_dataset = valid_data.map(preprocess_function, batched=True, remove_columns=['id', 'type', 'question', 'text'])
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(
        valid_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.valid_batch_size
    )

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss()
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * args.train_epochs),
        num_training_steps=(len(train_dataloader) * args.train_epochs),
    )
    metric = evaluate.load("accuracy")
    
    model.to(args.device)
    total_step=0
    for epoch in range(args.train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(args.device)
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels'].unsqueeze(1)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not loss:
                loss = loss_fn(outputs.logits, batch["labels"])
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            run.log(step=total_step, data={
                "step":total_step,
                "loss":loss,
                "learning rate":lr_scheduler.get_last_lr()[0],
            })
            total_step+=step
        
        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(args.device)
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels'].unsqueeze(1)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            predictions = outputs.logits.argmax(dim=-1)
            # predictions, references = predictions, labels
            
            
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        run.log(data={
                "eval/epoch":epoch,
                "eval/accuracy":eval_metric,
            })
        print(f"epoch {epoch}:", eval_metric)

    
    model.push_to_hub(repo_id, safe_serialization=args.safe_serialization, token=HF_W_TOKEN)
    tokenizer.push_to_hub(repo_id)
    