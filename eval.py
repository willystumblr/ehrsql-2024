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
from sft import add_default_args, set_seed, update_args, build_dataset
import torch
import argparse
from utils.prompt import create_eval_prompt_batch, create_prompt, create_sample_prompt
from utils.data_io import write_json as write_label
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
from scoring_program.reliability_score import calculate_score, penalize
from scoring_program.postprocessing import post_process_sql
from utils.settings import huggingface_login



parser = argparse.ArgumentParser()
parser = add_default_args(parser)
args = parser.parse_args()
# Determine device for training and set model save path
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.n_gpu = torch.cuda.device_count()
# Set random seed for reproducibility
set_seed(args)
train_data, valid_data, test_data = build_dataset()
# CKPT_PATH = "" # path/to/checkpoint


# """ TODO: Fix this part... """
huggingface_login()

model_config = dict(
    device_map={"":PartialState().local_process_index},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if args.bf16 else "auto",
    use_cache=False,
)


model = AutoModelForCausalLM.from_pretrained(args.model_name, config=model_config)
model = PeftModel.from_pretrained(model, args.load_checkpoint_path)


if args.train_type=='PPO':
    model.merge_and_unload()
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model, model_config)

tokenizer = AutoTokenizer.from_pretrained(args.load_checkpoint_path, padding_side='left') # since we added several tokens to the original tokenizer



def generate_sql(model, tokenizer, test_dataset, args, gen_config=None):
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
        prompts = create_eval_prompt_batch(batch)
        # Tokenize prompts for model input
        if args.test_batch_size<=1:
            inputs = tokenizer(prompts, return_tensors="pt")['input_ids']
            
        else:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)['input_ids']
        
        if args.device == 'cuda':
            inputs = inputs.cuda()

        # Generate predictions with inference mode for efficiency
        with torch.inference_mode():
            generated_outputs = model.generate(
                input_ids=inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **gen_config
            )

        # Move the generated sequences to the CPU if using CUDA.
        preds = generated_outputs["sequences"].cpu() if args.device == "cuda" else generated_outputs["sequences"]

        # Process logits and calculate probabilities and entropies.
        logits = torch.stack(generated_outputs["scores"], dim=1)[:: int(args.num_beams / args.num_samples)]
        logits = logits.cpu() if args.device == "cuda" else logits
        probs = torch.softmax(logits, dim=2).float()
        log_probs = torch.log_softmax(logits, dim=2).float()
        entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()


        # Determine if the current batch is for testing or training.
        is_test = True
        if "label" in batch:
            is_test = False
            reals = batch["label"]


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

        pred_list = tokenizer.batch_decode(preds[:, inputs.shape[1]:], skip_special_tokens=True)
        # print(pred_list)

        # Construct the output results for each prediction.
        for idx in range(len(preds)):
            result = {
                "id": batch['id'][idx],
                "question": batch['question'][idx],
                "pred": pred_list[idx],
                "entropy": entropy_list[idx],
            }

            # Include the real target output if it's training data.
            if not is_test:
                result["real"] = batch['label'][idx]

            results.append(result)
        # print(pred_list[idx])

    return results

def get_threshold(id2maxent, score_dict):
    """
    Determine the optimal threshold for filtering based on maximum entropy and scores.
    """
    values = []
    scores = []
    for key, val in id2maxent.items():
        values.append(val)
        scores.append(score_dict[key])

    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_scores = np.array(scores)[sorted_indices]

    max_score, threshold = 0, -1
    for idx in range(len(sorted_scores)):
        cum_score = sum(sorted_scores[:idx+1])
        if cum_score > max_score:
            print('cum_score > max_score')
            max_score, threshold = cum_score, sorted_values[idx-1]

    return threshold  # We abstain if maxent is greater than this threshold.


# Perform inference on the validation set
valid_eval = generate_sql(model, tokenizer, valid_data, args)

# Post-process SQL queries for evaluation
label = {sample['id']: post_process_sql(sample['real']) for sample in valid_eval}
label_y = {sample['id']: post_process_sql(sample['pred']) for sample in valid_eval}
id2maxent = {sample['id']: max(sample['entropy']) for sample in valid_eval}  # NOTE: Abstain strategy not used here

# Calculate the Reliability Score (RS) across all queries
real_dict = {id_: post_process_sql(label[id_]) for id_ in label}
pred_dict = {id_: post_process_sql(label_y[id_]) for id_ in label_y}
assert set(real_dict) == set(pred_dict), "IDs do not match"

scores, score_dict = calculate_score(real_dict, pred_dict, db_path=DB_PATH, return_dict=True)

accuracy0 = penalize(scores, penalty=0)
accuracy10 = penalize(scores, penalty=10)
accuracyN = penalize(scores, penalty=len(scores))

print(f"RS without filtering unanswerable queries: Accuracy0: {accuracy0}, Accuracy10: {accuracy10}, AccuracyN: {accuracyN}")

# Calculate threshold for filtering unanswerable queries
threshold = get_threshold(id2maxent, score_dict)
print(f"Threshold for filtering: {threshold}")

# Apply threshold to filter out uncertain predictions
label_y = {sample['id']: 'null' if threshold < max(sample['entropy']) else post_process_sql(sample['pred']) for sample in valid_eval}

# Recalculate RS with filtered predictions
real_dict = {id_: post_process_sql(label[id_]) for id_ in label}
pred_dict = {id_: post_process_sql(label_y[id_]) for id_ in label_y}

scores_filtered = calculate_score(real_dict, pred_dict, db_path=DB_PATH)

accuracy0_filtered = penalize(scores_filtered, penalty=0)
accuracy10_filtered = penalize(scores_filtered, penalty=10)
accuracyN_filtered = penalize(scores_filtered, penalty=len(scores))

# Output the refined RS scores with abstention
print(f"RS with filtered unanswerable queries: Accuracy0: {accuracy0_filtered}, Accuracy10: {accuracy10_filtered}, AccuracyN: {accuracyN_filtered}")

##### Submission #####
test_eval = generate_sql(model, tokenizer, test_data, args)

label_y = {sample['id']: 'null' if threshold < max(sample['entropy']) else post_process_sql(sample['pred']) for sample in test_eval}
from utils.data_io import write_json as write_label

# Save the filtered predictions to a JSON file
run_name = args.load_checkpoint_path.split("/")[-1]
os.makedirs(os.path.join(RESULT_DIR, run_name), exist_ok=True)

SCORING_OUTPUT_DIR = os.path.join(os.path.join(RESULT_DIR, run_name), 'prediction.json')
write_label(SCORING_OUTPUT_DIR, label_y)

# Verify the file creation
print(f"Listing files in RESULT_DIR: {os.listdir(RESULT_DIR)}")
