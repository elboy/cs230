import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

class BaseballNNDataset(Dataset):

    def __init__(self, df_split, df_full, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_split = df_split
        self.df_split = self.df_split.reset_index(drop=True)
        self.df_full = df_full
        self.transform = transform

    def __len__(self):
        return self.df_split.shape[0]

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        features = self.df_split.iloc[idx][2:-1].values.astype(np.float32)
        labels = np.array(self.df_split.loc[idx, 'label'])
        player_id = self.df_split.loc[idx, 'playerID']
        year = self.df_split.loc[idx, 'yearID']
        sample = (features, labels.reshape(1), player_id, year)
        return sample


class BaseballL5NNDataset(Dataset):

    def __init__(self, df_split, df_full, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_split = df_split
        self.df_split = self.df_split.reset_index(drop=True)
        self.df_full = df_full
        self.transform = transform

    def __len__(self):
        return self.df_split.shape[0]

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        labels = np.array(self.df_split.loc[idx, 'label'])
        player_id = self.df_split.loc[idx, 'playerID']
        year = self.df_split.loc[idx, 'yearID']

        player_prev_seasons = self.df_full[(self.df_full['playerID'] == player_id) & self.df_full['yearID'].isin(range(year - 5 + 1, year + 1))]


        next_fill = player_prev_seasons.loc[player_prev_seasons['yearID'].idxmax()]
        for year_index in range(year, year - 5, -1):
            row = player_prev_seasons[player_prev_seasons['yearID'] == year_index]
            if row.empty:
                next_fill['yearID'] = year_index
                player_prev_seasons = player_prev_seasons.append(next_fill)
            else:
                next_fill = row

        player_prev_seasons = player_prev_seasons.sort_values(by='yearID', ascending=True)
        features = np.array(player_prev_seasons.values)[:, 2:-1].flatten().astype(np.float32)

        sample = (features, labels.reshape(1), player_id, year)
        return sample

def load_dataset(args, dset_ext, last_k=False):
    # Get paths for dataset
    print("Dowmloading path: {}".format(os.path.join(os.getcwd(), args.data_dir, dset_ext)))
    dset_path = os.path.join(os.getcwd(), args.data_dir, dset_ext)
    path_df = os.path.join(os.getcwd(), args.data_dir, 'data.pkl')
 

    # Create the input data pipeline
    print("Creating the datasets...")
    dset_df = pd.read_pickle(dset_path)
    df = pd.read_pickle(path_df)

    if last_k:
        dset = BaseballL5NNDataset(dset_df, df)
    else:
        dset = BaseballNNDataset(dset_df, df)



    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dataset_size = len(dset)
    print(dataset_size, "rows")
    return dataloader, dataset_size