import os

BASE_DIR = '/Users/msk/project_repo/ehrsql-2024/'

DB_ID = 'mimic_iv'
BASE_DATA_DIR = f'{BASE_DIR}/data/{DB_ID}'
BASE_CKPT_DIR = f'{BASE_DIR}/ckpt'
if not os.path.exists(BASE_CKPT_DIR):
    os.makedirs(BASE_CKPT_DIR)
RESULT_DIR = f'{BASE_DIR}/result_submission/'
SCORE_PROGRAM_DIR = f'{BASE_DIR}/scoring_program/'

# File paths for the dataset and labels
TABLES_PATH = os.path.join(BASE_DATA_DIR, 'tables.json')               # JSON containing database schema
TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train', 'data.json')    # JSON file with natural language questions for training data
TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'train', 'label.json')  # JSON file with corresponding SQL queries for training data
VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, 'valid', 'data.json')    # JSON file for validation data
DB_PATH = os.path.join(BASE_DATA_DIR, f'{DB_ID}.sqlite')               # Database path
