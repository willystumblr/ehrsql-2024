import wandb
from huggingface_hub import login
import os, glob, json, argparse
from ast import literal_eval
from functools import partial
from tqdm.auto import tqdm
from pathlib import Path
import torch
import evaluate
from transformers import GenerationConfig
from transformers.integrations import WandbCallback

# Fill your own tokens
API = ""
HF_TOKEN = ''

def wandb_setup(key = API):
    wandb.login(key=API)
    
def huggingface_login(key=HF_TOKEN):
    login(token=HF_TOKEN)


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation", "target"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["query"]
            generation = self.generate(prompt=prompt)
            target = example["target"]
            records_table.add_data(prompt, generation, target, *list(self.gen_config.to_dict().values()))
        return records_table

    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})