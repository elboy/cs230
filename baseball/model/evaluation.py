"""Tensorflow utility functions for evaluation"""

import logging
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def test(model, criterion, dataloader, dataset_size, device):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """

    model.eval()
    y_true = []
    y_pred = []
    player_ids_list = []
    years_list = []

    test_loss = 0.0

    with torch.no_grad():
        # player_id is tuple of strings
        for inputs, labels, player_ids, years in dataloader:
            # might need this for gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.float()
            years = years.float()
            player_ids_list += list(player_ids)

            y_true += labels.cpu().numpy().squeeze().tolist()
            years_list += years.numpy().squeeze().tolist()


            preds = model(inputs)
            y_pred += preds.cpu().numpy().squeeze().tolist()
            loss = criterion(preds, labels)
            test_loss += loss



    test_loss = test_loss / dataset_size
    return (y_true, y_pred, player_ids_list, years_list, test_loss)


def rnn_test(model, criterion, dataloader, dataset_size, device):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """

    model.eval()
    y_true = []
    y_pred = []
    player_ids_list = []
    years_list = []

    test_loss = 0.0

    with torch.no_grad():
        # player_id is tuple of strings
        for inputs, labels, player_ids, years in dataloader:
            # might need this for gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.float()
            years = years.float()
            player_ids_list += list(player_ids)

            y_true += np.reshape(labels.cpu().numpy(), -1).tolist()
            years_list += np.reshape(years.numpy(), -1).tolist()


            preds = model(inputs)
            y_pred += np.reshape(preds.cpu().numpy(), -1).tolist()
            loss = criterion(preds, labels)
            test_loss += loss



    test_loss = test_loss / dataset_size
    return (y_true, y_pred, player_ids_list, years_list, test_loss)