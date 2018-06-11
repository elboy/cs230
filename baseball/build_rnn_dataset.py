"""Read, split and save the kaggle dataset for our model"""
import pandas as pd
import numpy as np
import pickle

path_dataset = 'data/kaggle/data.pkl'
# Load the dataset into memory
print("Loading Kaggle dataset into memory...")
DF = pd.read_pickle(path_dataset)
print("Parsed {} player seasons.".format(DF.shape[0]))
print("- done.")

if __name__ == "__main__":

    rnn_dset = {}
    print("Iterating through data...")

    for i, row in DF.iterrows():
        if i % 500 == 0:
            print("Completed {} player seasons.".format(i))
        player_id = row['playerID']
        year_id = row['yearID']

        player_prev_seasons = DF[(DF['playerID'] == player_id) & (DF['yearID'] <= year_id)]
        player_prev_seasons = player_prev_seasons.sort_values(by='yearID', ascending=True)
        features = np.array(player_prev_seasons.values)[:, 2:-1].astype(np.float32)
        print(features.shape)

        rnn_dset[(player_id, year_id)] = features

    print("Parsed {} player seasons.".format(len(rnn_dset)))
    print("- done.")
    pickle.dump( rnn_dset, open( "data/kaggle/rnn.pkl", "wb" ) )