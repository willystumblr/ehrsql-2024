from peft import PeftConfig, PeftModel
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
from peft import LoraConfig, PeftConfig, PeftModel
from eval_multi import get_threshold
from utils.args import add_default_args
import torch
import argparse
from utils.prompt import create_eval_prompt_batch, create_pipeline_prompt, create_prompt, create_sample_prompt
from utils.data_io import write_json as write_label, build_dataset
import pandas as pd
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_io import (
    RESULT_DIR,
    DB_PATH,
)
from accelerate import PartialState, Accelerator
from torch.utils.data import DataLoader
import torch
from scoring_program.scoring_utils import execute_all, reliability_score, penalize
from scoring_program.postprocessing import post_process_sql
from utils.settings import huggingface_login, set_seed
import logging
from unsloth import FastLanguageModel
from model.pipleline import Model
from eval_pipeline import generate_sql

def generate_sql(model, tokenizer, tokenizer_cls, test_dataset, args, gen_config=None):
    """
    Generate SQL queries from a test dataset using a causal language model in batches.

    Parameters:
    - model: AutoModelForCausalLM, the trained model for generating text.
    - tokenizer: Tokenizer, the tokenizer for encoding and decoding texts.
    - test_dataset: datasets.Dataset, the dataset containing test samples.
    - args: Namespace, containing configuration like device, batch size, etc.
    - gen_config: dict, the configuration for the generation process.

    Returns:
    A list of dictionaries with generated SQL queries and additional information.
    """
    model.eval()
    model.to(args.device)
    results = []

    if not gen_config: 
        gen_config = dict(
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            min_length=-1, # don't ignore the EOS token (see above)
            top_k=0.0, # no top-k sampling
            top_p=1.0, # no nucleus sampling
            do_sample=True, # yes, we want to sample
            pad_token_id=tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
            
        )
                        

    # Create DataLoader for batch processing
    dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Iterate over batches
    for batch in tqdm(dataloader):
        
        # Format each item in the batch to create prompts
        cls_prompts, gen_prompts = create_pipeline_prompt(batch)
        # print(cls_prompts)
        # Tokenize prompts for model input
        if args.test_batch_size<=1:
            cls_inputs = tokenizer_cls(cls_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_seq_length)
            gen_tokenized = tokenizer(gen_prompts, return_tensors="pt")
            gen_inputs=gen_tokenized['input_ids']
            attention_mask = gen_tokenized['attention_mask']
            
        else:
            cls_inputs = tokenizer_cls(cls_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
            gen_tokenized = tokenizer(cls_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)['input_ids']
            gen_inputs=gen_tokenized['input_ids']
            attention_mask = gen_tokenized['attention_mask']
            
        if args.device == 'cuda':
            gen_inputs, attention_mask = gen_inputs.cuda(), attention_mask.cuda()

        # Generate predictions with inference mode for efficiency
        with torch.inference_mode():
            generated_outputs = model.generate(
                cls_input_ids=cls_inputs,
                gen_input_ids=gen_inputs,
                attention_mask=attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                gen_config=gen_config
            )
        
        
        assert len(generated_outputs) == len(batch['id']), AssertionError(f"len(generated_outputs): {len(generated_outputs)}, len(batch): {len(batch)}")
        
        for output in generated_outputs:
            # print(output['type'])
            if output['type'] == "text2sql":
                # Move the generated sequences to the CPU if using CUDA.
                preds = output["sequences"].cpu() if args.device == "cuda" else output["sequences"]

                # Process logits and calculate probabilities and entropies.
                logits = torch.stack(output["scores"], dim=1)[:: int(args.num_beams / args.num_samples)]
                logits = logits.cpu() if args.device == "cuda" else logits
                probs = torch.softmax(logits, dim=2).float()
                log_probs = torch.log_softmax(logits, dim=2).float()
                entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()


            # Initialize lists to store predictions, probabilities, and entropies.

                entropy_list = []

                # Process each prediction in the batch.
                for idx in range(len(preds)):

                    pred_tensor = preds[idx][1:]
                    entropy_truncated = entropies[idx].tolist()

                    # Truncate the prediction at the end-of-sequence token, if present.
                    if tokenizer.eos_token_id in pred_tensor:
                        pred_eos_idx = torch.nonzero(pred_tensor == tokenizer.eos_token_id)[0].item()
                        entropy_truncated = entropy_truncated[: pred_eos_idx + 1]

                    entropy_list.append(entropy_truncated)

                pred = tokenizer.batch_decode(preds[:, gen_inputs.shape[1]:], skip_special_tokens=True)
            elif output['type'] == 'answerability':
                # print(output)
                # pred_list = []
                # entropy_list = []
                logits = output.logits.cpu() if args.device == "cuda" else output.logits
                preds = logits.argmax(-1).tolist()
                # assert generated_outputs.logits.argmax(-1).items() == 0, AssertionError("Something went wrong! Classifier and Generator didn't work properly.")
                assert preds[0]==0, AssertionError("Something went wrong! Classifier and Generator didn't work properly.")
                pred = ['null\n']
                entropy_list = [[0.0004147880245000124, 5.0285681936657056e-05, 3.618660002757679e-07]]
            else:
                raise NotImplementedError("Error occurred!")
            # Construct the output results for each prediction; assuming test batch size is always 1
            result = {
                "id": batch['id'][0],
                "question": batch['question'][0],
                "pred": str(pred[0]),
                "entropy": entropy_list[0],
            }

            # print(output.keys)
            # Determine if the current batch is for testing or training.
            not_test = "label" in batch or "labels" in batch
            if not_test:
                # is_test = False
                try:
                    result["real"] = batch["label"][0]
                except Exception as e:
                    result["real"] = batch["labels"][0]
                
                if result['real'] == 'null':
                    print(result)
            
            results.append(result)

    return results

def null_accuracy(label_dict:dict, pred_dict:dict):
    nulls = []
    for k in label_dict.keys():
        if label_dict[k]=='null':
            nulls.append(k)
    null_hit = list(filter(lambda k: pred_dict[k]=='null' and k in nulls, pred_dict.keys()))
    
    return len(null_hit)/len(nulls)


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
    # Set random seed for reproducibility
    set_seed(args)
    train_data, valid_data, test_data = build_dataset(args)
    # CKPT_PATH = "" # path/to/checkpoint


    # """ TODO: Fix this part... """
    huggingface_login()

    
    model = Model(args.model_name, args.model_name_2, args=args)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_2)
    tokenizer_cls = AutoTokenizer.from_pretrained(args.model_name)
    # Perform inference on the validation set
    valid_eval = generate_sql(model, tokenizer, tokenizer_cls, valid_data, args)

    # Post-process SQL queries for evaluation
    label = {sample['id']: post_process_sql(sample['real']) for sample in valid_eval}
    label_y = {sample['id']: post_process_sql(sample['pred']) for sample in valid_eval}
    id2maxent = {sample['id']: max(sample['entropy']) for sample in valid_eval}  # NOTE: Abstain strategy not used here

    # Calculate the Reliability Score (RS) across all queries
    real_dict = {id_: post_process_sql(label[id_]) for id_ in label}
    pred_dict = {id_: post_process_sql(label_y[id_]) for id_ in label_y}
    assert set(real_dict) == set(pred_dict), "IDs do not match"

    real_result = execute_all(real_dict, db_path=DB_PATH, tag='real')
    pred_result = execute_all(pred_dict, db_path=DB_PATH, tag='pred')

    scores, score_dict = reliability_score(real_result, pred_result, return_dict=True)
    accuracy0 = penalize(scores, penalty=0)
    accuracy5 = penalize(scores, penalty=5)
    accuracy10 = penalize(scores, penalty=10)
    accuracyN = penalize(scores, penalty=len(scores))

    logger.info(f"*** RS without filtering unanswerable queries: Accuracy0: {accuracy0}, Accuracy5: {accuracy5}, Accuracy10: {accuracy10}, AccuracyN: {accuracyN} ***")
    logger.info(f"*** Null hit Accuracy: {null_accuracy(real_dict, pred_dict)} ***")
    # Calculate threshold for filtering unanswerable queries
    threshold = get_threshold(id2maxent, score_dict)
    logger.info(f"Threshold for filtering: {threshold}")

    # Apply threshold to filter out uncertain predictions
    label_y = {sample['id']: 'null' if threshold < max(sample['entropy']) else post_process_sql(sample['pred']) for sample in valid_eval}

    # Recalculate RS with filtered predictions
    real_dict = {id_: post_process_sql(label[id_]) for id_ in label}
    pred_dict = {id_: post_process_sql(label_y[id_]) for id_ in label_y}

    scores_filtered = reliability_score(real_dict, pred_dict)

    accuracy0_filtered = penalize(scores_filtered, penalty=0)
    accuracy5_filtered = penalize(scores_filtered, penalty=5)
    accuracy10_filtered = penalize(scores_filtered, penalty=10)
    accuracyN_filtered = penalize(scores_filtered, penalty=len(scores))

    # Output the refined RS scores with abstention
    logger.info(f"*** RS with filtered unanswerable queries: Accuracy0: {accuracy0_filtered}, Accuracy5: {accuracy5_filtered}, Accuracy10: {accuracy10_filtered}, AccuracyN: {accuracyN_filtered} ***")
    logger.info(f"*** Null hit Accuracy: {null_accuracy(real_dict, pred_dict)} ***")
    
