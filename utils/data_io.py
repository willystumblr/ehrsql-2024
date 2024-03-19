import os
import json
from datasets import Dataset

BASE_DIR = '.'

DB_ID = 'mimic_iv'
BASE_DATA_DIR = f'{BASE_DIR}/data/{DB_ID}'
BASE_CKPT_DIR = f'{BASE_DIR}/ckpt/ehrsql-2024'
RESULT_DIR = f'{BASE_DIR}/result_submission/'
SCORE_PROGRAM_DIR = f'{BASE_DIR}/scoring_program/'

# File paths for the dataset and labels
TABLES_PATH = os.path.join(BASE_DATA_DIR, 'tables.json')                # JSON containing database schema
TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, 'train') # JSON file with fetched answers for eah SQL query for training data
VALID_DATA_DIR = os.path.join(BASE_DATA_DIR, 'valid')     # JSON file for validation data
TEST_DATA_DIR = os.path.join(BASE_DATA_DIR, 'test')
DB_PATH = os.path.join(BASE_DATA_DIR, f'{DB_ID}.sqlite')                # Database path

# Will be deprecated during test phase
NEW_TRAIN_DIR = os.path.join(BASE_DATA_DIR, '__train')
NEW_VALID_DIR = os.path.join(BASE_DATA_DIR, '__valid')
NEW_TEST_DIR = os.path.join(BASE_DATA_DIR, 'valid')


def read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file

def write_json(path, file, mode='w+'):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, mode) as f:
        json.dump(file, f)

def build_dataset(args):
    # prepare data
    if args.phase == 'dev':
        train_path, valid_path, test_path = NEW_TRAIN_DIR, NEW_VALID_DIR, NEW_TEST_DIR  
    elif args.phase == 'dev_final':
        train_path, valid_path, test_path = TRAIN_DATA_DIR, VALID_DATA_DIR, None
    else:
        train_path, valid_path, test_path = TRAIN_DATA_DIR, VALID_DATA_DIR, TEST_DATA_DIR
    
    
    new_train_data = read_json(os.path.join(train_path, 'data.json'))
    new_train_label = read_json(os.path.join(train_path, "label.json"))
    # new_train_answer = read_json(os.path.join(NEW_TRAIN_DIR, "answer.json"))
    if args.train_type=='text2sql':
        train_dataset = [{"id": d['id'], "type":'text2sql',"question":d['question'], "label":l[1]} for d, l in zip(new_train_data['data'], new_train_label.items())]
    elif args.train_type=='unanswerable':
        train_dataset = []
        for d, l in zip(new_train_data['data'], new_train_label.items()):
            example = {"id": d['id'], "type":'unanswerable',"question":d['question']}
            example['label']=True if l[1] =='null' else False
            train_dataset.append(example)
    else:
        raise ValueError("Unsupported train_type: should be either 'text2sql' or 'unanswerable'.")

    train_data = Dataset.from_list(train_dataset)
    ### TODO: leave it for now..
    if args.phase=='dev':
        new_valid_data = read_json(os.path.join(valid_path, 'data.json'))
        new_valid_label = read_json(os.path.join(valid_path, "label.json"))
        if args.train_type=='text2sql':
            valid_dataset = [{"id": d['id'], "question":d['question'], "label":l[1]} for d, l in zip(new_valid_data['data'], new_valid_label.items())]
        elif args.train_type=='unanswerable':
            valid_dataset = []
            for d, l in zip(new_valid_data['data'], new_valid_label.items()):
                example = {"id": d['id'], "type":'unanswerable',"question":d['question']}
                example['label']=True if l[1] =='null' else False
                valid_dataset.append(example)
        else:
            raise ValueError("Unsupported train_type: should be either 'text2sql' or 'unanswerable'.")
        
        valid_data = Dataset.from_list(valid_dataset)

        
        new_test_data = read_json(os.path.join(NEW_TEST_DIR, "data.json"))
        test_dataset = [{"id": d['id'], "question":d['question']} for d in new_test_data['data']]
        test_data = Dataset.from_list(test_dataset)
    elif args.phase=='dev_final':
        valid_data=None
        test_data=None 
    else:
        pass
    
    return train_data, valid_data, test_data
