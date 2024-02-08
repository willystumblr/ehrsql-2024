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
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.tools_setup import wandb_setup, huggingface_login, LLMSampleCB
from peft import LoraConfig # get_peft_model
from trl.trainer import SFTTrainer
from transformers import TrainingArguments
from utils.files import (
    BASE_DATA_DIR,
    BASE_CKPT_DIR,
    TABLES_PATH,
    TRAIN_DATA_PATH,
    TRAIN_LABEL_PATH,
    VALID_DATA_PATH,
    DB_PATH,
    NEW_TRAIN_DIR,
    NEW_VALID_DIR,
    NEW_TEST_DIR
)

from utils.prompt import create_eval_prompt_batch, create_prompt, create_sample_prompt
from utils.data_io import read_json as read_data
from utils.data_io import write_json as write_data



wandb_setup()
huggingface_login()
WANDB_PROJ_NAME = 'ehrsql-2024-sft'
CKPT_OUT_DIR = f'{BASE_CKPT_DIR}/sft'


def add_default_args(parser):
    """
    Define and set default arguments for the script.
    """
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--db_id", type=str, default="mimiciii", help="database name")  # NOTE: `mimic_iv` will be used for codabench
    parser.add_argument("--train_data_dir", type=str, help="train data path")
    parser.add_argument("--valid_data_dir", type=str, help="valid data path")
    parser.add_argument("--test_data_dir", type=str, help="test data path")
    parser.add_argument("--tables_file", type=str, help="table schema path")
    parser.add_argument("--is_saved_ckpt", type=bool, default=False, help="set True when ppo, dpo, or eval.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    parser.add_argument("--output_file", type=str, default="prediction_raw.json", help="output file name")
    parser.add_argument("--train_type", type=str, default="sft", help="train type; either sft, ppo, or dpo")

    # basic parameters
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_checkpoint_path", type=str, default=None)
    parser.add_argument("--load_checkpoint_path", type=str, default=None)

    # training parameters
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=str, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--evaluation_strategy", type=str, default='no')
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)

    # lora parameters
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=int, default=0.1)
    parser.add_argument("--lora_r", type=int, default=1)


    parser.add_argument("--save_every_epoch", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # generation parameters
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser

def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def update_args(new_args, prev_args):
    """
    Update training arguments with the values saved in the checkpoint.
    """
    for arg in vars(prev_args):
        if arg not in new_args:
            setattr(new_args, arg, getattr(prev_args, arg))
    return new_args

def build_dataset():
    # prepare data
    new_train_data = read_data(os.path.join(NEW_TRAIN_DIR, 'data.json'))
    new_train_label = read_data(os.path.join(NEW_TRAIN_DIR, "label.json"))
    new_valid_data = read_data(os.path.join(NEW_VALID_DIR, 'data.json'))
    new_valid_label = read_data(os.path.join(NEW_VALID_DIR, "label.json"))
    new_test_data = read_data(os.path.join(NEW_TEST_DIR, "data.json"))

    train_dataset = [{"id": d['id'], "question":d['question'], "label":l[1]} for d, l in zip(new_train_data['data'], new_train_label.items())]
    valid_dataset = [{"id": d['id'], "question":d['question'], "label":l[1]} for d, l in zip(new_valid_data['data'], new_valid_label.items())]
    test_dataset = [{"id": d['id'], "question":d['question']} for d in new_test_data['data']]

    train_data = Dataset.from_list(train_dataset)
    valid_data = Dataset.from_list(valid_dataset)
    test_data = Dataset.from_list(test_dataset)
    
    return train_data, valid_data, test_data
    
    
parser = argparse.ArgumentParser()
parser = add_default_args(parser)
args = parser.parse_args()
# Determine device for training and set model save path
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.n_gpu = torch.cuda.device_count()
args.output_dir = CKPT_OUT_DIR = f'{BASE_CKPT_DIR}/{args.train_type}'
# Set random seed for reproducibility
set_seed(args)


# WandB setting
os.environ["WANDB_PROJECT"] = WANDB_PROJ_NAME  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

# Configure CUDA settings
# This code is originally written for Google Colab
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_data, valid_data, test_data = build_dataset()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
add_tokens = ["<", "<=", "<>"]
tokenizer.add_tokens(add_tokens)

peft_parameters = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"]
)


model_config = dict(
    device_map={"":0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

os.makedirs({args.output_dir}/{wandb.run.name}, exist_ok=True)
training_args = TrainingArguments(
    output_dir = f'{args.output_dir}/{wandb.run.name}',
    report_to="wandb", # enables logging to W&B 😎
    per_device_train_batch_size=args.train_batch_size,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    lr_scheduler_type=args.lr_scheduler_type,
    num_train_epochs=args.train_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps, # simulate larger batch sizes
    bf16=args.bf16,
    evaluation_strategy=args.evaluation_strategy,
    save_strategy=args.evaluation_strategy, # if load_best_model_at_end=True
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    load_best_model_at_end=args.load_best_model_at_end
)

trainer = SFTTrainer(
    model=args.model_name,
    model_init_kwargs=model_config,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=valid_data,
    packing=True, # pack samples together for efficient training
    max_seq_length=args.max_seq_length, # maximum packed length
    args=training_args,
    peft_config=peft_parameters,
    formatting_func=create_prompt, # format samples with a model schema
)

sample_dataset = valid_data.map(create_sample_prompt)
wandb_callback = LLMSampleCB(trainer, sample_dataset, num_samples=20, max_new_tokens=args.max_new_tokens)
trainer.add_callback(wandb_callback)
trainer.train()
torch.cuda.empty_cache()
