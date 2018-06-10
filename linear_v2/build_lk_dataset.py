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

def create_last5(player_id, year_id):
    player_prev_seasons = DF[(DF['playerID'] == player_id) & DF['yearID'].isin(range(year_id - 5 + 1, year_id + 1))]

    next_fill = player_prev_seasons.loc[player_prev_seasons['yearID'].idxmax()]
    for year_index in range(year_id, year_id - 5, -1):
        row = player_prev_seasons[player_prev_seasons['yearID'] == year_index]
        if row.empty:
            next_fill['yearID'] = year_index
            player_prev_seasons = player_prev_seasons.append(next_fill)
        else:
            next_fill = row

    player_prev_seasons = player_prev_seasons.sort_values(by='yearID', ascending=True)
    features = np.array(player_prev_seasons.values)[:, 2:-1].flatten().astype(np.float32)
    return features

if __name__ == "__main__":

    last5_dset = {}
    print("Iterating through data...")

    for i, row in DF.iterrows():
        if i % 500 == 0:
            print("Completed {} player seasons.".format(i))
        player_id = row['playerID']
        year_id = row['yearID']

        last5 = create_last5(player_id, year_id)
        last5_dset[(player_id, year_id)] = last5

    print("Parsed {} player seasons.".format(len(last5_dset)))
    print("- done.")
    pickle.dump( last5_dset, open( "data/kaggle/last5.pkl", "wb" ) )
    