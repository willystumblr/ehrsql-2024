import json
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
import os
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.tools_setup import wandb_setup, huggingface_login
from peft import LoraConfig, PeftConfig, PeftModel
from sft import add_default_args, set_seed, update_args, build_dataset
import torch
import argparse
from utils.prompt import create_eval_prompt_batch, create_prompt, create_sample_prompt
import sqlite3
import pandas as pd
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
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

BASE_MODEL_DIR = f'{BASE_CKPT_DIR}/SFT'
WANDB_PROJ_NAME = 'ehrsql-2024-ppo'
parser = argparse.ArgumentParser()
parser = add_default_args(parser)
args = parser.parse_args()
# Determine device for training and set model save path
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.n_gpu = torch.cuda.device_count()
args.output_dir = f'{BASE_CKPT_DIR}/{args.train_type}'
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


num_ppo_sample = 400
train_sample = train_data.select(random.sample(range(len(train_data)), num_ppo_sample))
ppo_dataset = train_sample.map(create_sample_prompt)
ppo_dataset = ppo_dataset.remove_columns(["id", "question"])

def reward_model(sql_file_path, csv_dir_path, target_query, pred_query):
    # Connect to a database (or create one if it doesn't exist)
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Read and execute the SQL schema from the .sql file
    try:
        with open(sql_file_path, 'r') as sql_file:
            sql_script = sql_file.read()
        cursor.executescript(sql_script)
        conn.commit()
    except Exception as e:
        print(f"Failed to execute schema script: {e}")
        conn.close()
        raise  # Error encountered

    # Import CSV files into the database
    for csv_file in os.listdir(csv_dir_path):
        if csv_file.endswith('.csv'):
            try:
                table_name = os.path.splitext(csv_file)[0]
                df = pd.read_csv(os.path.join(csv_dir_path, csv_file))
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            except Exception as e:
                print(f"Failed to import {csv_file}: {e}")
                conn.close()
                raise  # Error encountered

    if target_query=='null':
        return 1.0 if pred_query=='null' else -10.0

    # Execute the target query to get the expected output
    try:
        cursor.execute(target_query)
        expected_output = cursor.fetchall()  # This will serve as the expected output for comparison
        conn.commit()
    except Exception as e:
        print(f"Failed to execute target query: {e}")
        conn.close()
        raise  # Target query unexecutable

    # Execute the prediction query
    try:
        cursor.execute(pred_query)
        pred_output = cursor.fetchall()  # This is the prediction output for comparison
        conn.commit()
    except Exception as e:
        print(f"Failed to execute prediction query: {e}")
        conn.close()
        return -5.0  # Prediction query unexecutable

    # Close the connection
    conn.close()

    # Compare the prediction output with the expected output
    if pd.DataFrame(expected_output).equals(pd.DataFrame(pred_output)):
        return 1.0  # Prediction is correct
    else:
        return 0.0  # Prediction is wrong but executable

# Assume that we load saved checkpoint after SFT.
ppo_config = PPOConfig(
    log_with ='wandb',
    tracker_project_name=args.project_name,
    learning_rate=1.41e-5,
    ppo_epochs=args.train_epochs,
    batch_size=args.train_batch_size,
)


# Same config with SFT
model_config = dict(
    device_map={"":0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

peft_parameters = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"]
)


ckpt_path = f"" # contains both adapter config and tokenizer config

tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(args.model_name, config=model_config)
model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=True)

model.merge_and_unload()

model = AutoModelForCausalLMWithValueHead.from_pretrained(model, model_config)

ppo_trainer = PPOTrainer(
    model=model,
    ref_model=args.model_name,
    config=ppo_config,
    dataset=ppo_dataset,
    tokenizer=tokenizer
)


generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": args.max_new_tokens, # specify how many tokens you want to generate at most
}


sql_file_path = f"{BASE_DATA_DIR}/mimic_iv.sql"
csv_dir_path = f"{BASE_DATA_DIR}"
save_path = f"{args.output_dir}/{wandb.run.name}"

for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        tokenized_queries = tokenizer(batch['query'], return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)['input_ids'].cuda()
        query_tensors = [tokenized_queries[i] for i in range(len(tokenized_queries))]
        targets = batch['label']

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, batch_size=ppo_trainer.config.batch_size, return_prompt=False, **generation_kwargs)
        batch['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True) #


        #### Compute reward score
        rewards = [torch.tensor(reward_model(sql_file_path, csv_dir_path, t, p)) for t, p in zip(targets, batch['response'])]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        

os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)