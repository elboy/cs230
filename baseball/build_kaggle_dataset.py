"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
import random
import pandas as pd
import numpy as np


def save_dataset(dataset, save_dir):
    """Writes features.npy and labels.npy files in save_dir from dataset"""
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    features = dataset[:, :-1]
    np.save(os.path.join(save_dir, 'features.npy'), features)
    labels = dataset[:, -1:]
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    # np.load(outfile) to load
    print("- done.")


def load_dataset(path_csv, at_bats_threshold=0, filter_null=True):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    df = pd.read_csv(path_csv)
    rows, cols = df.shape
    print("Raw data has {} rows and {} cols". format(rows, cols))

    if filter_null:
        # axis=0 drops rows
        df.dropna(axis=0, inplace=True)
        rows, cols = df.shape
        print("Data after omitting NaN has {} rows and {} cols". format(rows, cols))

    stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
    df = df.groupby(['playerID', 'yearID'])[stats].sum()
    rows, cols = df.shape
    print("Data after combinging stints has {} rows and {} cols". format(rows, cols))

    df = df[df.AB >= at_bats_threshold]
    rows, cols = df.shape
    print("Data after filtering by AB threshold has {} rows and {} cols". format(rows, cols))

    return df


def add_labels(df):
    hrs = df['HR'].values
    hrs = np.append(hrs[1:], -1)

    df = df.assign(label=hrs)

    for (player_id, year), row in df.iterrows():
        if (player_id, year + 1) not in df.index:
            df.loc[(player_id, year), 'label'] = -1

    # Omit seasons where the player doesn't play the next season
    df = df[df.label != -1]
    rows, cols = df.shape
    print("Data with good labels has {} rows and {} cols". format(rows, cols))
    
    return df.reset_index()


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/kaggle/Batting.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg
    
    at_bats_threshold = 100

    # Load the dataset into memory
    print("Loading Kaggle dataset into memory...")
    df = load_dataset(path_dataset, at_bats_threshold, True)
    df = add_labels(df)
    print("Parsed {} player seasons.".format(df.shape[0]))
    print("- done.")
    
    # 42 for jackie robinson
    #random.seed(42)
    #random.shuffle(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the dataset into train, dev and split (dummy split with no shuffle)
    df_len = df.shape[0]
    train_df = df[:int(0.8*df_len)]
    dev_df = df[int(0.8*df_len) : int(0.9*df_len)]
    test_df = df[int(0.9*df_len):]

    print(train_df)

    #train_df.to_pickle('data/kaggle/train.pkl')
    #dev_df.to_pickle('data/kaggle/dev.pkl')
    #test_df.to_pickle('data/kaggle/test.pkl')
    #df.to_pickle('data/kaggle/data.pkl')

    # Save the datasets to files
    #save_dataset(train_dataset, 'data/kaggle/train')
    #save_dataset(dev_dataset, 'data/kaggle/dev')
    #save_dataset(test_dataset, 'data/kaggle/test')
    