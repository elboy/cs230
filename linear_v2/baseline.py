"""Run a baseline model"""
import argparse
import os
import sys
import pandas as pd 
import numpy as np
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


arg_parser = argparse.ArgumentParser(description="parser for baseball project")
arg_parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")



arg_parser.add_argument("--name", type=str, default=None,
                              help="name for the model")
arg_parser.add_argument("--years", type=int, default=5, help="average over past years years")
arg_parser.add_argument("--verbose", action="store_true")
args = arg_parser.parse_args()

if __name__ == '__main__':
    print("Downloading datasets from eval path: {}".format(os.path.join(os.getcwd(), args.data_dir, 'dev.pkl')))
    path_eval = os.path.join(os.getcwd(), args.data_dir, 'dev.pkl')
    path_df = os.path.join(os.getcwd(), args.data_dir, 'data.pkl')
 

    # Create the input data pipeline
    print("Creating the datasets...")
    eval_df = pd.read_pickle(path_eval)
    df = pd.read_pickle(path_df)
    eval_df = eval_df.reset_index(drop=True)
    print("Dataset shape: ", eval_df.shape)
    print("- done.")

    # Run the model
    print("Starting evaluation")
    y_true = []
    y_pred = []
    for index, row in eval_df.iterrows():
        y_true.append(row['label'])

        player_id = row['playerID']
        year = row['yearID']
        player_prev_seasons = df[(df['playerID'] == player_id) & df['yearID'].isin(range(year - args.years + 1, year + 1))]
        pred = player_prev_seasons['HR'].mean()
        y_pred.append(pred)

    print(len(y_true))
    print(len(y_pred))
    r2 = metrics.r2_score(y_true, y_pred)
    print(r2)


