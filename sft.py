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
from utils.settings import set_seed, wandb_setup, huggingface_login, LLMSampleCB, HF_W_TOKEN
from peft import LoraConfig # get_peft_model
from trl.trainer import SFTTrainer
# from trl.trainer import get_kbit_device_map, get_quantization_config
from transformers import TrainingArguments
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

"""
python sft.py \
    --train_type=SFT \
    --project_name=ehrsql-2024-sft \
    --model_name=meta-llama/Llama-2-7b-hf \
    --train_epochs=3 \
    --train_batch_size=8 \
    --valid_batch_size=4 \
    --learning_rate=1e-3 \
    --logging_steps=10 \
    --lr_scheduler_type=cosine \
    --bf16=1 \
    --db_id=mimic_iv \
    --evaluation_strategy=epoch \
    --test_batch_size=1 \
    --save_strategy=epoch \
    --load_best_model_at_end=True
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

    train_data, valid_data, test_data = build_dataset(args.phase)

    ### WandB setting
    wandb_setup(args)
    huggingface_login()

    os.environ["WANDB_PROJECT"] =  args.project_name # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Configure CUDA settings
    # This code is originally written for Google Colab
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))


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
        device_map={"": Accelerator().process_index},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        use_cache=False,
    )
    save_dir = f"{args.output_dir}/{wandb.run.name}"
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, model_config)
    
    os.makedirs(save_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir = save_dir,
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
        load_best_model_at_end=args.load_best_model_at_end,
        logging_first_step=args.logging_first_step,
        push_to_hub=True,
        push_to_hub_model_id=f"{args.project_name}-{args.model_name.split('/')[-1]}",
        push_to_hub_token=HF_W_TOKEN
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data if args.phase == 'dev' else None,
        packing=True, # pack samples together for efficient training
        max_seq_length=args.max_seq_length, # maximum packed length
        args=training_args,
        peft_config=peft_parameters,
        formatting_func=create_prompt, # format samples with a model schema
    )

    if args.phase != 'dev':
        sample_dataset = valid_data.map(create_sample_prompt)
        wandb_callback = LLMSampleCB(trainer, sample_dataset, num_samples=20, max_new_tokens=args.max_new_tokens)
        trainer.add_callback(wandb_callback)
    
    trainer.train()
    torch.cuda.empty_cache()
    trainer.model = trainer.accelerator.unwrap_model(trainer.model)
    trainer.push_to_hub()
    #trainer.push_to_hub()

