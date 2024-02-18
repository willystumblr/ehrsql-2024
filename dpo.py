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
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.settings import set_seed, wandb_setup, huggingface_login, LLMSampleCB
from peft import LoraConfig # get_peft_model
from trl.trainer import SFTTrainer
from transformers import TrainingArguments
from utils.data_io import (
    BASE_DATA_DIR,
    BASE_CKPT_DIR,
    NEW_TRAIN_DIR,
    NEW_VALID_DIR
)
from utils.args import add_default_args
from utils.prompt import create_eval_prompt_batch, create_prompt, create_sample_prompt
from utils.data_io import read_json as read_data
from utils.data_io import write_json as write_data
from utils.data_io import build_dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer
from accelerate import PartialState

"""
python dpo.py \
    --project_name=ehrsql-2024-dpo \
    --train_type=DPO \
    --bf16=1 \
    --load_checkpoint_path=/path/to/ckpt \
    --train_batch_size=8 \
    --valid_batch_size=4 \
    --logging_steps=10 \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --load_best_model_at_end=True \
    --train_epochs=3
"""
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args = parser.parse_args()
    # Determine device for training and set model save path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.output_dir = f'{BASE_CKPT_DIR}/{args.train_type.lower()}'

    # Set random seed for reproducibility
    set_seed(args)

    wandb_setup(args)
    huggingface_login()

    os.environ["WANDB_PROJECT"] = args.project_name  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Configure CUDA settings
    # This code is originally written for Google Colab
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    ### load dataset
    dpo_train_data = read_data(f"{NEW_TRAIN_DIR}/dpo_data.json")
    dpo_valid_data = read_data(f"{NEW_VALID_DIR}/dpo_data.json")

    dpo_train_set = Dataset.from_list(dpo_train_data).rename_column("query", "prompt")
    dpo_valid_set = Dataset.from_list(dpo_valid_data).rename_column("query", "prompt")


    model_config = dict(
        device_map={"":PartialState().local_process_index},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        use_cache=False,
        load_in_4bit=True
    )

    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args = parser.parse_args()
    # Determine device for training and set model save path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.output_dir = f'{BASE_CKPT_DIR}/{args.train_type.lower()}'

    # Set random seed for reproducibility
    set_seed(args)


    ckpt_path = args.load_checkpoint_path # SFT checkpoint; contains both adapter config and tokenizer config
    model, tokenizer = FastLanguageModel.from_pretrained(ckpt_path, config=model_config)


    training_args = TrainingArguments(
        output_dir = os.path.join(args.output_dir, wandb.run.name), # should be sth like ehrsql-2024/DPO/{wandb.run.name}
        report_to="wandb", # enables logging to W&B 😎
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps, # simulate larger batch sizes
        bf16=args.bf16,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy, # if load_best_model_at_end=True
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        logging_first_step=args.logging_first_step
    )


    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dpo_train_set,
        tokenizer=tokenizer,
        eval_dataset=dpo_valid_set, 
    )

    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

