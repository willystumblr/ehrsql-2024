import json
import numpy as np
import pandas as pd
from collections import Counter
from datasets import Dataset
import os
from tqdm.auto import tqdm
import wandb
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.args import add_default_args
from utils.settings import set_seed, wandb_setup, huggingface_login
from peft import LoraConfig, PeftConfig, PeftModel

import torch
import argparse
from utils.prompt import create_eval_prompt_batch, create_ppo_prompt, create_prompt, create_sample_prompt
import sqlite3
import pandas as pd
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_io import (
    BASE_DATA_DIR,
    BASE_CKPT_DIR,
    build_dataset,
)
import time
from accelerate import Accelerator
import logging
import sqlparse
"""
python ppo.py \
    --train_type=PPO \
    --project_name=ehrsql-2024-ppo \
    --train_epochs=3 \
    --train_batch_size=4 \
    --model_name=meta-llama/Llama-2-7b-hf \
    --learning_rate=1e-3 \
    --load_checkpoint_path=/path/to/adapter \
    --bf16=1 \
    --num_samples=400
"""


def reward_model(sql_file_path, csv_dir_path, target_query, pred_query):
    # Connect to a database (or create one if it doesn't exist)
    db_exists = False
    if os.path.exists('mimic_iv_demo.db'):
        db_exists = True
    conn = sqlite3.connect(f'mimic_iv_demo.db')
    conn.execute("PRAGMA journal_mode = wal")
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
    if not db_exists:
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
        return 1.0 if pred_query=='null' else -1.0

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
        return -1.0  # Prediction query unexecutable

    # Close the connection
    conn.close()

    # Compare the prediction output with the expected output
    if pd.DataFrame(expected_output).equals(pd.DataFrame(pred_output)):
        return 1.0  # Prediction is correct
    else:
        return 0.0  # Prediction is wrong but executable

def reward_model_v2(sql_file_path, target_query, pred_query):
    """_summary_ 
    Rule-based reward model. Reward is determined by the followings:
    - executable
    - correctness
    """ 
    ### null handling
    if target_query=='null':
        return 1.0 if pred_query=='null' else -1.0

    valid = syntax_checker(pred_query)
    
    if valid:
        target_output = execute_query(sql_file_path, target_query)
        pred_output = execute_query(sql_file_path, pred_query)
        if target_output==pred_output:
            return 1.0
        else:
            return 0.5
    else:
        return -1.0

def syntax_checker(query):
    valid = False
    temp_db = sqlite3.connect(":memory:")   
    cursor = temp_db.cursor()
    try:
        _ = cursor.execute(query)
    except sqlite3.OperationalError as e:
        error_message = str(e).lower()
        if 'unrecognized token:' in error_message or 'syntax error' in error_message:
            pass
        elif 'no such' in error_message:
            valid = True
            return valid
    finally:
        temp_db.close()
        

def execute_query(sql_file_path, query):
    conn = sqlite3.connect(sql_file_path)
    cursor = conn.cursor()
    output = cursor.execute(query)
    rows = output.fetchall()
    conn.close()
    return rows

if __name__=="__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args = parser.parse_args()
    # Determine device for training and set model save path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.output_dir = f'{BASE_CKPT_DIR}/{args.train_type}'
    # Set random seed for reproducibility
    set_seed(args)

    # WandB & Huggingface setting
    wandb_setup(args)
    huggingface_login()

    os.environ["WANDB_PROJECT"] = args.project_name  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Configure CUDA settings
    # This code is originally written for Google Colab
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(args.n_gpu))

    train_data, valid_data, test_data = build_dataset()


    logger.info("*** Sampling PPO Datasets... ***")
    train_sample = train_data.select(random.sample(range(len(train_data)), args.num_samples))
    null_count = len(list(filter(lambda x: x['label']=='null', train_sample)))
    while (null_count > args.num_samples*0.2) or (null_count < args.num_samples*0.05):
        train_sample = train_data.select(random.sample(range(len(train_data)), args.num_samples))
    
    ppo_dataset = train_sample.map(create_ppo_prompt)
    # ppo_dataset = ppo_dataset.rename_column("question", "query")
    ppo_dataset = ppo_dataset.remove_columns(["id"])

    # Assume that we load saved checkpoint after SFT.
    ppo_config = PPOConfig(
        log_with ='wandb',
        tracker_project_name=args.project_name,
        learning_rate=1.41e-5,
        ppo_epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        mini_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # accelerator_kwargs={"num_processes":args.n_gpu}
    )


    # Same config with SFT
    model_config = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
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

    ### NOTE: args.load_checkpoint_path contains both adapter config and tokenizer config!
    logger.info("*** Loading checkpoints ***")
    tokenizer = AutoTokenizer.from_pretrained(args.load_checkpoint_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=model_config)
    model = PeftModel.from_pretrained(model, args.load_checkpoint_path, is_trainable=True)

    model.merge_and_unload()
    """
    from: https://github.com/huggingface/trl/issues/1036

    @younesbelkada commented on Dec 4, 2023
    hi @Reza-esfandiarpoor
    Technically yes, but I would advise to first merge the sft_model into a single base model and pass the merged model to AutoModelForCausalLMWithValueHead.
    """

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model, model_config)
    
    ### initialize reference model
    if not model.is_peft_model:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, )
        ref_model = PeftModel.from_pretrained(ref_model, args.load_ref_checkpoint_path, is_trainable=False) # fresse Peft
        ref_model.merge_and_unload()
        
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model, model_config)
    else:
        ref_model = None # Default value

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
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


    sql_file_path = f"{BASE_DATA_DIR}/mimic_iv.sqlite"
    csv_dir_path = f"{BASE_DATA_DIR}"
    save_path = f"{args.output_dir}/{wandb.run.name}"

    logger.info(f"*** NUM_DEVICES: {ppo_trainer.accelerator.num_processes} ***")
    logger.info("*** START TRAINING ***")
    for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "process: "):
        for batch in tqdm(ppo_trainer.dataloader):
            tokenized_queries = tokenizer(batch['query'], return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)['input_ids'].cuda()
            query_tensors = [tokenized_queries[i] for i in range(len(tokenized_queries))]
            targets = batch['label']

            # print(targets)
            #### Get response from SFTModel
            response_tensors, ref_response_tensors = ppo_trainer.generate(query_tensors, batch_size=ppo_trainer.config.batch_size, generate_ref_response=True, return_prompt=False, **generation_kwargs)
            batch['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True) #
            # print(batch['response'])
            # print(batch['ref_response'])
            
            #### Compute reward score
            # rewards = [torch.tensor(reward_model(sql_file_path, csv_dir_path, t, p)) for t, p in zip(targets, batch['response'])]
            # ref_rewards = [torch.tensor(reward_model(sql_file_path, csv_dir_path, t, p)) for t,  p in zip(targets, batch['ref_response'])]
            rewards = [torch.tensor(reward_model_v2(sql_file_path, t, p)) for t, p in zip(targets, batch['response'])]
            ref_rewards = [torch.tensor(reward_model_v2(sql_file_path, t, p)) for t, p in zip(targets, batch['ref_response'])]
            
            batch["ref_rewards"] = ref_rewards

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
            

    os.makedirs(save_path, exist_ok=True)
    ppo_trainer.save_pretrained(save_path)
