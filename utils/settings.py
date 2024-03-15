import wandb
from huggingface_hub import login
import os, glob, json, argparse
# from ast import literal_eval
from functools import partial
from tqdm.auto import tqdm
# from pathlib import Path
import torch
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
import numpy as np
import random

# Fill your own tokens
API = "e845b522dd2be52f09a0b6b36051a1007fb1bda7"
HF_TOKEN = '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'
HF_W_TOKEN = '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'

PADDING_MAP = {
    'google/gemma-2b-it':'right',
    'meta-llama/Llama-2-7b-hf':'left',
}

def wandb_setup(args: argparse.Namespace, key = API):
    wandb.login(key=API)
    # os.makedirs(wandb_path)
    wandb.init(project=args.project_name, dir=args.wandb_dir)
    
def huggingface_login(key=HF_W_TOKEN):
    login(token=HF_W_TOKEN)

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


class LLMSampleCB(WandbCallback):
    """
    Reference: 
    https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy
    """
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            try:
                output = self.model.generate(inputs=tokenized_prompt['input_ids'], generation_config=self.gen_config)
            except:
                output = self.model.generate(**tokenized_prompt, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation", "target"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["question"]
            generation = self.generate(prompt=prompt)
            target = example["label"]
            records_table.add_data(prompt, generation, target, *list(self.gen_config.to_dict().values()))
        return records_table

    def on_save(self, args, state, control,  **kwargs):
        super().on_save(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})
