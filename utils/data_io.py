import os
import json
from datasets import Dataset

BASE_DIR = '.'

DB_ID = 'mimic_iv'
BASE_DATA_DIR = f'{BASE_DIR}/data/{DB_ID}'
BASE_CKPT_DIR = f'{BASE_DIR}/ckpt/ehrsql-2024'
if not os.path.exists(BASE_CKPT_DIR):
    os.makedirs(BASE_CKPT_DIR)
RESULT_DIR = f'{BASE_DIR}/result_submission/'
SCORE_PROGRAM_DIR = f'{BASE_DIR}/scoring_program/'

# File paths for the dataset and labels
TABLES_PATH = os.path.join(BASE_DATA_DIR, 'tables.json')                # JSON containing database schema
TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train', 'data.json')     # JSON file with natural language questions for training data
TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'train', 'label.json')   # JSON file with corresponding SQL queries for training data
TRAIN_ANSWER_PATH = os.path.join(BASE_DATA_DIR, 'train', 'answer.json') # JSON file with fetched answers for eah SQL query for training data
VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, 'valid', 'data.json')     # JSON file for validation data
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

def build_dataset():
    # prepare data
    new_train_data = read_json(os.path.join(NEW_TRAIN_DIR, 'data.json'))
    new_train_label = read_json(os.path.join(NEW_TRAIN_DIR, "label.json"))
    new_train_answer = read_json(os.path.join(NEW_TRAIN_DIR, "answer.json"))
    new_valid_data = read_json(os.path.join(NEW_VALID_DIR, 'data.json'))
    new_valid_label = read_json(os.path.join(NEW_VALID_DIR, "label.json"))
    new_test_data = read_json(os.path.join(NEW_TEST_DIR, "data.json"))

    train_dataset = [{"id": d['id'], "question":d['question'], "label":l[1], "answer":a[1]} for d, l, a in zip(new_train_data['data'], new_train_label.items(), new_train_answer.items())]
    valid_dataset = [{"id": d['id'], "question":d['question'], "label":l[1]} for d, l in zip(new_valid_data['data'], new_valid_label.items())]
    test_dataset = [{"id": d['id'], "question":d['question']} for d in new_test_data['data']]

    train_data = Dataset.from_list(train_dataset)
    valid_data = Dataset.from_list(valid_dataset)
    test_data = Dataset.from_list(test_dataset)
    
    return train_data, valid_data, test_data
