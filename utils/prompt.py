def create_prompt(example):
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
    prompt=None
    if example['label']=='' or 'label' not in example.keys():
        prompt= ("Convert the question below to SQL query."
                "\n\n"
                "### Question:\n{question}\n\n### SQL:\n").format_map(example)
    else:
        prompt= ("Convert the question below to SQL query."
                "\n\n"
                "### Question:\n{question}\n\n### SQL:\n{label}").format_map(example)
    return prompt


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
    prompts = []

    questions = batch['question']

    for q in questions:
        prompt= ("Convert the question below to SQL query."
                "\n\n"
                "### Question:\n{}\n\n### SQL:\n").format(q)
        prompts.append(prompt)

    return prompts

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
    return {"query":create_prompt(example), "label":label, "answer":example['answer']}
