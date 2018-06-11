"""Train the model"""
import argparse
import logging
import os
import sys
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
#from model.utils import Params
#from model.utils import set_logger
from model.training import train
from model.evaluation import test
from model.input_fn import load_dataset
from model.model_fn import BaseballFCN
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


arg_parser = argparse.ArgumentParser(description="parser for baseball project")
arg_parser.add_argument('--model_dir', default='models/lastk', help="Directory to store best model weights")
arg_parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")
#parser.add_argument('--restore_dir', default=None, help="Optional, directory containing weights to reload before training")



arg_parser.add_argument("--name", type=str, default=None,
                              help="name for the model")
arg_parser.add_argument("--k-years", type=int, default=5,
                              help="how many years to look back")
arg_parser.add_argument("--epochs", type=int, default=50,
                              help="number of training epochs, default is 10")
arg_parser.add_argument("--lr", type=float, default=1e-4,
                              help="learning rate")
arg_parser.add_argument("--weight-decay", type=float, default=0,
                              help="weight decay")
arg_parser.add_argument("--hidden-size", type=int, default=200,
                              help="number of hidden units")
arg_parser.add_argument("--batch-size", type=int, default=32,
                              help="batch size")
arg_parser.add_argument("--log-epoch-interval", type=int, default=20,
                              help="how often to update epochs")
arg_parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
arg_parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
#arg_parser.add_argument("--preload-model", type=str, default=None, help="directory of stored model")
arg_parser.add_argument("--verbose", action="store_true")
args = arg_parser.parse_args()


def train_and_evaluate(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # Set the logger
    # set_logger(os.path.join(args.model_dir, 'train.log'))
    print("Downloading datasets")
    train_dl, train_sz = load_dataset(args, 'train.pkl', True, args.k_years)
    dev_dl, dev_sz = load_dataset(args, 'dev.pkl', True, args.k_years)
    print("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    print("Creating the model...")
    model = BaseballFCN(17 * args.k_years, args.hidden_size).to(device)
    if use_cuda:
        model = model.cuda()
    print("- done.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    best_model_wts = model.state_dict()
    best_r2 = float("-inf")
    #best_y_pred = []
    #best_y_true = []
    losses = {"train": [], "val": []}
    r2s = {"train": [], "val": []}

    # Train the model
    print("Starting training for {} epoch(s)".format(args.epochs))
    since = time.time()
    #best_model = train_and_evaluate(model, criterion, optimizer, args, dataloaders, dataset_sizes, use_cuda)
    for epoch in range(1, args.epochs + 1):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, args.epochs))
        print(time.ctime())
        

        model, y_true, y_pred, train_loss = train(model, criterion, optimizer, args, train_dl, train_sz, device)
        epoch_r2 = metrics.r2_score(y_true, y_pred)
        print("{} Loss: {:.4f} R2: {:.4f}".format("train", train_loss, epoch_r2))
        losses["train"].append(train_loss)
        r2s["train"].append(epoch_r2)

        y_true, y_pred, player_ids, years, test_loss = test(model, criterion, dev_dl, dev_sz, device)

        epoch_r2 = metrics.r2_score(y_true, y_pred)
        losses["val"].append(test_loss)
        r2s["val"].append(epoch_r2)
        print("{} Loss: {:.4f} R2: {:.4f}".format("eval", test_loss, epoch_r2))
        if epoch_r2 > best_r2:
            #best_y_true = y_true
            #best_y_pred = y_pred
            best_r2 = epoch_r2
            best_model_wts = model.state_dict()


    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
    time_elapsed // 60, time_elapsed % 60))
    print("Best R2: {:4f}".format(best_r2))

    pickle.dump( losses, open( "model/last5_losses.pkl", "wb" ) )
    pickle.dump( r2s, open( "model/last5_r2s.pkl", "wb" ) )

    model.cpu()
    model.load_state_dict(best_model_wts)
    model_name = "best_lastk.model"
    print("Model name: ", model_name)
    save_model_path = os.path.join(args.model_dir, model_name)
    torch.save(model.state_dict(), save_model_path)
    return best_r2


if __name__ == '__main__':
    train_and_evaluate(args)
    #learning_rates = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    #hidden_units = [50, 100, 200, 500]
    #k_years = [1, 2, 3, 4, 5]
    #weight_decays = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    #r2s = []
    #for wd in weight_decays:
    #    print("Weight decays:", wd)
    #    args.weight_decay = wd
    #    best_r2 = train_and_evaluate(args)
    #    r2s.append(best_r2)

    #print("Wegiht decays:")
    #print(weight_decays)
    #print("R^2:")
    #print(r2s)







