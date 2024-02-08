import json
import numpy as np
import pandas as pd
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from utils.files import (
    DB_ID,
    BASE_DATA_DIR,
    BASE_CKPT_DIR,
    TABLES_PATH,
    TRAIN_DATA_PATH,
    TRAIN_LABEL_PATH,
    VALID_DATA_PATH,
    DB_PATH
)


from utils.data_io import read_json as read_data
from utils.data_io import write_json as write_data

# Load train and validation sets
train_data = read_data(TRAIN_DATA_PATH)
train_label = read_data(TRAIN_LABEL_PATH)
valid_data = read_data(VALID_DATA_PATH)



# Define stratification criteria for consistent distribution between answerable and unanswerable questions
stratify = ['unans' if train_label[id_]=='null' else 'ans' for id_ in list(train_label.keys())]

# Split the original training data into new training and validation sets, while maintaining the distribution
new_train_keys, new_valid_keys = train_test_split(
    list(train_label.keys()),
    train_size=0.8,
    random_state=42,
    stratify=stratify
)

# Initialize containers for the new training and validation sets
new_train_data = []
new_train_label = {}
new_valid_data = []
new_valid_label = {}

# Sort each sample into the new training or validation set as determined by the split
for sample in train_data['data']:
    if sample['id'] in new_train_keys:
        new_train_data.append(sample)
        new_train_label[sample['id']] = train_label[sample['id']]
    elif sample['id'] in new_valid_keys:
        new_valid_data.append(sample)
        new_valid_label[sample['id']] = train_label[sample['id']]
    else:
        # If a sample is neither in the train nor valid keys, raise an error
        raise ValueError(f"Error: Sample with ID {sample['id']} has an invalid split.")

# Structure the new datasets in a JSON-compatible format
new_train_data = {'version': f'{DB_ID}_sample', 'data': new_train_data}
new_valid_data = {'version': f'{DB_ID}_sample', 'data': new_valid_data}

# Set directory for the new splitted data
NEW_TRAIN_DIR = os.path.join(BASE_DATA_DIR, '__train')
NEW_VALID_DIR = os.path.join(BASE_DATA_DIR, '__valid')
NEW_TEST_DIR = os.path.join(BASE_DATA_DIR, 'valid')

# Save the new datasets to JSON files for later use
write_data(os.path.join(NEW_TRAIN_DIR, "data.json"), new_train_data)
write_data(os.path.join(NEW_TRAIN_DIR, "label.json"), new_train_label)
write_data(os.path.join(NEW_VALID_DIR, "data.json"), new_valid_data)
write_data(os.path.join(NEW_VALID_DIR, "label.json"), new_valid_label)