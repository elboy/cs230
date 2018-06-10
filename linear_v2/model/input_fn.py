import os
import pickle
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


class BaseballLKNNDataset(Dataset):

    def __init__(self, df_split, k, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_split = df_split
        self.df_split = self.df_split.reset_index(drop=True)
        self.last5_dset = pickle.load( open( "data/kaggle/last5.pkl", "rb" ) )
        self.k = k
        self.transform = transform

    def __len__(self):
        return self.df_split.shape[0]

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        labels = np.array(self.df_split.loc[idx, 'label'])
        player_id = self.df_split.loc[idx, 'playerID']
        year = self.df_split.loc[idx, 'yearID']

        features = self.last5_dset[(player_id, year)]
        features = features[(5 - self.k) * 17 :]

        sample = (features, labels.reshape(1), player_id, year)
        return sample

class BaseballRNNDataset(Dataset):

    def __init__(self, df_split, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_split = df_split
        self.df_split = self.df_split.reset_index(drop=True)
        self.rnn_dset = pickle.load( open( "data/kaggle/rnn.pkl", "rb" ) )
        self.transform = transform

    def __len__(self):
        return self.df_split.shape[0]

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        labels = np.array(self.df_split.loc[idx, 'label'])
        player_id = self.df_split.loc[idx, 'playerID']
        year = self.df_split.loc[idx, 'yearID']

        features = self.rnn_dset[(player_id, year)]

        sample = (features, labels.reshape(1), player_id, year)
        return sample



def load_dataset(args, dset_ext, last_k=False, k=None, rnn=False):
    # Get paths for dataset
    print("Dowmloading path: {}".format(os.path.join(os.getcwd(), args.data_dir, dset_ext)))
    dset_path = os.path.join(os.getcwd(), args.data_dir, dset_ext)
    path_df = os.path.join(os.getcwd(), args.data_dir, 'data.pkl')
 

    # Create the input data pipeline
    print("Creating the datasets...")
    dset_df = pd.read_pickle(dset_path)
    df = pd.read_pickle(path_df)

    if last_k:
        dset = BaseballLKNNDataset(dset_df, k)
    elif rnn:
        dset = BaseballRNNDataset(dset_df)
    else:
        dset = BaseballNNDataset(dset_df, df)


    if rnn:
        dataloader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=1)
    else: 
        dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dataset_size = len(dset)
    print(dataset_size, "rows")
    print(" - done.")
    return dataloader, dataset_size