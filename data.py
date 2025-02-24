from datasets import load_dataset
import re
import numpy as np
import pandas as pd
import os

# For each language, sampling 5000 from each language as training data, 1k as dev/test data respectively
TRAIN_SAMPLE_PER_LANGAUGE = 5000
VALIDATE_SAMPLE_PER_LANGAUGE = 625
TEST_SAMPLE_PER_LANGAUGE = 625

def load_data():
    # Load the Wiki40B dataset (from on Hugging Face)
    # The selected language in the classification task for 10 languages:
    # English, German, French, Arabic, Russian, Korean, Chinese
    X_train = []
    y_train = []
    X_validation = []
    y_validation = []
    X_test = []
    y_test = []

    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path to the directory
    dir_path = os.path.join(current_dir, 'dataset')

    # Check if the dataset directory exists
    if os.path.isdir(dir_path):
        print(f"The dataset dir '{dir_path}' exists.")
        X_train = (pd.read_csv(os.path.join(dir_path, 'imdb_train.csv')).T)
        X_validation = (pd.read_csv(os.path.join(dir_path, 'imdb_validation.csv')).T)
        X_test = (pd.read_csv(os.path.join(dir_path, 'imdb_test.csv')).T)
        y_train = (pd.read_csv(os.path.join(dir_path, 'imdb_y_train.csv')).T)
        y_test = (pd.read_csv(os.path.join(dir_path, 'imdb_y_test.csv')).T)
        y_validation = (pd.read_csv(os.path.join(dir_path, 'imdb_y_validation.csv')).T)

    else:
        print(f"The dataset dir '{dir_path}' does not exist. Loading data from hugging face.")
        os.makedirs(dir_path, exist_ok=True)
        language_label = ['en', 'de', 'fr', 'ar', 'ru', 'ko', 'zh-cn']
        
        for lang in language_label:
            dataset = load_dataset("wiki40b", lang)
            x_train, x_validation, x_test = split_data(dataset)
            X_train.extend(x_train)
            X_validation.extend(x_validation)
            X_test.extend(x_test)

            y_train.extend([lang] * TRAIN_SAMPLE_PER_LANGAUGE)
            y_validation.extend([lang] * VALIDATE_SAMPLE_PER_LANGAUGE)
            y_test.extend([lang] * VALIDATE_SAMPLE_PER_LANGAUGE)

        # Save the train split to a CSV file
        X_train = pd.DataFrame(X_train).T
        X_train.to_csv(os.path.join(dir_path, 'imdb_train.csv'), index=False)
            
        # Save the validation split to a CSV file
        X_validation = pd.DataFrame(X_validation).T
        X_validation.to_csv(os.path.join(dir_path, 'imdb_validation.csv'), index=False)
            
        # Save the test split to a CSV file
        X_test = pd.DataFrame(X_test).T
        X_test.to_csv(os.path.join(dir_path,'imdb_test.csv'), index=False)
            
        # Save the train split to a CSV file
        y_train = pd.DataFrame(y_train).T
        y_train.to_csv(os.path.join(dir_path,'imdb_y_train.csv'), index=False)
            
        # Save the train split to a CSV file
        y_validation = pd.DataFrame(y_validation).T
        y_validation.to_csv(os.path.join(dir_path,'imdb_y_validation.csv'), index=False)
            
        # Save the train split to a CSV file
        y_test = pd.DataFrame(y_test).T
        y_test.to_csv(os.path.join(dir_path,'imdb_y_test.csv'), index=False)

    print(f"loaded dataset size: training {X_train.shape}, testing {X_test.shape}")
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def split_data(dataset):
    # sample the dataset
    train_data = preprocess_data(dataset['train']['text']).sample(TRAIN_SAMPLE_PER_LANGAUGE).tolist()
    dev_data = preprocess_data(dataset['validation']['text']).sample(VALIDATE_SAMPLE_PER_LANGAUGE).tolist()
    test_data = preprocess_data(dataset['test']['text']).sample(TEST_SAMPLE_PER_LANGAUGE).tolist()
    return train_data, dev_data, test_data

def preprocess_data(dataset):
    # remove the special markers
    pd_text = pd.DataFrame(dataset, columns=['text'])
    pd_text = pd_text['text'].str.replace(r'[!@#$(),"%^*?:;~`0-9]|_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_|_NEWLINE_|[\[\]]|\n', '', regex=True)
    return pd_text

if __name__ == "__main__":
    load_data()