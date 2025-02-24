from model import FastTextModel
from data import load_data
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import argparse

def create_checkpoint_path():
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path to the directory
    dir_path = os.path.join(current_dir, 'checkpoints')

    # Create the directory if not exists
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create the full path to the directory
    ckp_path = os.path.join(dir_path, 'cp.ckpt.weights.h5')
    return ckp_path
        
def vectorize_data(vectorize_layer, X_train, X_validation, X_test):
    X_train = vectorize_layer(X_train)
    X_validation = vectorize_layer(X_validation)
    X_test = vectorize_layer(X_test)

    vocab_size = len(vectorize_layer.get_vocabulary())
    print(f"Training data shape {X_train.shape}, Validation data shape {X_validation.shape}, Test data shape {X_test.shape}, vocabulary size: {vocab_size}")
    return X_train, X_validation, X_test, vocab_size

def encode_label(encoder, y_train, y_validation, y_test):
    # Convert y labels to one-hot encoder
    y_train = encoder.fit_transform(np.array(y_train))
    y_validation = encoder.fit_transform(np.array(y_validation))
    y_test = encoder.fit_transform(np.array(y_test))
    
    print(f"Training label shape {y_train.shape}, Validation label shape {y_validation.shape}, Test label shape {y_test.shape}")
    return y_train, y_validation, y_test

def parse_args():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="A command line tool to run language detection.")

    # Add command-line arguments
    parser.add_argument('--train', action='store_true', help="Whether to train the model on training dataset")
    parser.add_argument('--evaluate', action='store_true', help="Whether to run the trained model on a test dataset and obtain accuracy")
    parser.add_argument('--predict', type=str, help="run the model on a single input text and obtain detected language label")

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = load_data()
    # Convert the data to vectors
    vectorize_layer = tf.keras.layers.TextVectorization(output_mode='int', split='character', output_sequence_length=2500, ngrams=3)
    vectorize_layer.adapt(X_train)
    X_train, X_validation, X_test, vocab_size = vectorize_data(vectorize_layer, X_train, X_validation, X_test)

    # Convert the label to one-hot vectors
    encoder = OneHotEncoder(sparse_output=False)
    y_train, y_validation, y_test = encode_label(encoder, y_train, y_validation, y_test)
    print(y_validation)

    model =  FastTextModel(embedding_dim=128, num_lang=7, vocab_size=vocab_size, check_point_path=create_checkpoint_path())

    args = parse_args()
    if args.train:
        print(f"Training model")
        model.train(X_train, y_train, epoch=20, batch_size=500)
    
    if args.evaluate:
        print(f"Evaluating model")
        model.evaluate(X_test, y_test)

    if args.predict is not None:
        test_input = vectorize_layer(pd.DataFrame([args.predict]))
        model.predict(test_input, encoder)