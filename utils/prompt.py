# Reference:
# https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy
from .data_io import TABLES

def unanswerable_prompt(example):
    prompt_1= ("Given the tables and columns of the database, is the question convertible to an SQL query?"
                "\n\n"
                "### Database: \n\n"
                f"{str(TABLES)}"
                "\n\n")
    prompt_2=("### Question:\n{question}\n\n### Answer:{label}\n").format_map(example)
    return prompt_1+prompt_2
    
def text2sql_prompt(example):
    prompt_1 = ("Given the tables and columns of the database, convert the question below to SQL query if it is convertible."
                "\n\n"
                "### Database: \n\n"
                f"{str(TABLES)}"
                "\n\n")
    prompt_2 = ("### Question:\n{question}\n\n### Answer:{label}\n").format_map(example)
    return prompt_1+prompt_2

def create_prompt(example):
    prompt_formatter = unanswerable_prompt if example['type']=='unanswerable' else text2sql_prompt
    """_summary_
    formatting function for SFTTrainer;

    Args:
    - item: element of datasets.Dataset
        {
            "id": q_id,
            "question: query,
            "label": answer,
        }
    
    """
    # print(item)
    if 'label' not in example.keys():
        example['label'] = ''
    
    return prompt_formatter(example)


def create_eval_prompt_batch(batch):
    """_summary_
    create prompts during evaluation step, designed for batch processing.

    Args:
    - batch:
        {
            "id": [] # list of ids, len(batch)
            "question": [] # list of questions, len(batch)
            "label": [] # list of labels, len(batch) (optional for evaluation)
        }
    
    """
    task_type = batch['type'][0]
    examples = [{'question':q, 'type':task_type, 'label':''} for q in batch['question']]

    return [create_prompt(example) for example in examples]

def create_sample_prompt(example):
    """_summary_
    creates prompt for wandb logging.

    Args:
    - item: element of datasets.Dataset
        {
            "id": q_id,
            "question: query,
            "label": answer,
        }
    
    """
    label = example['label']
    example['label'] = ''
    return {"question":create_prompt(example), "label":label}


def create_ppo_prompt(example):
    """_summary_
    creates prompt for wandb logging.

    Args:
    - item: element of datasets.Dataset
        {
            "id": q_id,
            "question: query,
            "label": answer,
            "answer": sql query answer
        }
    
    """
    label = example['label']
    example['label'] = ''
    return {"query":create_prompt(example), "label":label}