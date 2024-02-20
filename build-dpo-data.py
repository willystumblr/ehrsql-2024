import gc
from tqdm.auto import tqdm
from utils.data_io import read_json as read_data, write_json as write_data
from utils.data_io import (
    DB_ID,
    NEW_TRAIN_DIR,
    NEW_VALID_DIR,
    NEW_TEST_DIR,
    BASE_CKPT_DIR,
    BASE_DATA_DIR
)
import os
from datasets import Dataset
from utils.prompt import create_eval_prompt_batch, create_prompt, create_sample_prompt
from utils.settings import huggingface_login, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.args import add_default_args
import torch
import argparse
import sqlite3
import pandas as pd
import os
from ppo import reward_model
from torch.utils.data import DataLoader
import random
import json


"""
python data/mimic_iv/build-dpo-data.py \
    --output_dir=data/mimic_iv \
    --load_checkpoint_path=/path/to/sft-checkpoint \
    --bf16=1 \
    --model_name=meta-llama/Llama-2-7b-hf \
    --train_batch_size=4 \
    --num_return_sequences=2 \
    --build_type=train

"""

def build_and_save(args, model, tokenizer, dataset, batch_size, num_return_sequences, save_path):
    if os.path.exists(save_path):
        processed = read_data(save_path)
        processed_data = Dataset.from_list(processed)
        dataset = dataset.filter(lambda x: x['id'] not in processed_data['id'])
        random_indices = random.sample(range(len(dataset)), 900)
        dataset = dataset.select(random_indices)
        mode = 'a'
    else:
        mode = 'r'
    predictions = build_dataset(model, tokenizer, dataset, batch_size, num_return_sequences)
    new_dataset = post_process(predictions)
    write_data(save_path, new_dataset, mode)


def build_dataset(model, tokenizer, dataset, batch_size, num_return_sequences):
    # dataset = dataset.rename_column("question", "query")
    # dataset = dataset.rename_column("label", "chosen")
    sample_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    torch.cuda.empty_cache()
    gc.collect()

    predictions = []

    gen_config = dict(
        max_new_tokens=512,
        num_beams=1,
        min_length=-1, # don't ignore the EOS token (see above)
        top_k=0.0, # no top-k sampling
        top_p=1.0, # no nucleus sampling
        do_sample=True, # yes, we want to sample
        pad_token_id=tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    )


    sql_file_path = f"{BASE_DATA_DIR}/mimic_iv.sql"
    csv_dir_path = f"{BASE_DATA_DIR}"

    model.to(args.device)
    model.eval()
    for batch in tqdm(sample_dataloader):

        example_prompts = create_eval_prompt_batch(batch)
        inputs = tokenizer(example_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)['input_ids'] 
        if args.device=="cuda":
            inputs = inputs.cuda()
        
        with torch.inference_mode():
            generated_outputs = model.generate(
                input_ids=inputs,
                output_scores=True,
                return_dict_in_generate=True,
                num_return_sequences=num_return_sequences,
                **gen_config
            )

        preds = generated_outputs["sequences"].cpu()
        pred_list = tokenizer.batch_decode(preds[:, inputs.shape[1]:], skip_special_tokens=True)

        batch_preds = []
        for b in range(0, len(pred_list), num_return_sequences):
            batch_preds.append(pred_list[b:b+num_return_sequences])
        # print(batch_preds)
        scores = []
        for i, pred in enumerate(batch_preds):
            ans=batch['label'][i]
            scores.append([reward_model(sql_file_path, csv_dir_path, ans, p) for p in pred])

        for i in range(len(batch['id'])):
            predictions.append({
                "id": batch['id'][i],
                "query": batch['question'][i],
                "chosen": batch['label'][i],
                "pred": batch_preds[i],
                "score": scores[i],
            })

    return predictions

def post_process(predictions):
    new_dpo_data = []
    indices = []
    for i, item in enumerate(predictions):
        processed = False
        if item['chosen']=='null':
            for p in item['pred']:
                if p != 'null':
                    rejected = p
                    processed = True
                    break
            if not processed:
                rejected = None
                print(f"{i}-th item needs to be processed!")
                indices.append(i)
        else:
            for p, s in zip(item['pred'], item['score']):
                if s < 1:
                    rejected = p
                    processed = True
                    break
            if not processed:
                rejected = 'null'
                print(f"{i}-th item was replaced with `null`!")

        new_dpo_data.append({
            "id": item['id'],
            "query": item['query'],
            "chosen": item['chosen'],
            "rejected": rejected,
        })

    random_preds = []
    for items in predictions:
        preds = [p for p in items['pred'] if p!='null']
        random_preds.extend(preds)

    for i in indices:
        assert new_dpo_data[i]['rejected'] is None and new_dpo_data[i]['chosen']=='null'
        new_dpo_data[i]['rejected'] = random.choice(random_preds)

    return new_dpo_data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args = parser.parse_args()
    # Determine device for training and set model save path
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Set random seed for reproducibility
    set_seed(args)

    new_data = None
    new_label = None
    
    if args.build_type == 'train':
    # Save the new datasets to JSON files for later use
        new_data = read_data(os.path.join(NEW_TRAIN_DIR, "data.json"))
        new_label = read_data(os.path.join(NEW_TRAIN_DIR, "label.json"))
    else:
        new_data = read_data(os.path.join(NEW_VALID_DIR, "data.json"))
        new_label = read_data(os.path.join(NEW_VALID_DIR, "label.json"))

    dataset = Dataset.from_list(
                        [{"id": d['id'], "question":d['question'], "label":l[1]} for d, l in zip(new_data['data'], new_label.items())]
                    )

    huggingface_login()


    model_config = dict(
        device_map={"":0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.load_checkpoint_path, padding_side='left')

    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=model_config)
    model = PeftModel.from_pretrained(model, args.load_checkpoint_path)

    dataset = dataset.map(create_sample_prompt) # .remove_columns(["question"]).rename_column("label", "chosen")
    
    build_and_save(args, model, tokenizer, dataset, args.train_batch_size, args.num_return_sequences, os.path.join(args.output_dir, f'__{args.build_type}', 'dpo_data.json'))
