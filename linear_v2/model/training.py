"""Tensorflow utility functions for training"""

import logging
import os
import time
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable


def train(model, criterion, optimizer, args, dataloader, dataset_size, device):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """

    model.train()

    y_true = []
    y_pred = []
    
    running_loss = 0.0

    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):
        # might need this for gpu
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()

        
        y_true += labels.cpu().numpy().squeeze().tolist()
        
        preds = model(inputs)
        y_pred += preds.cpu().detach().numpy().squeeze().tolist()
        
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.verbose:
            print("Batch", i, "Loss:", loss)

        running_loss += loss

    epoch_loss = running_loss / dataset_size
    return (model, y_true, y_pred, epoch_loss)

    """
    def save_logs(epoch_no=None):

        epoch_prefix = str(epoch_no) if epoch_no else ""
        if not os.path.exists(os.path.join(home_dir, "models/", model_name, epoch_prefix)):
            os.mkdir(os.path.join(home_dir, "models/", model_name, epoch_prefix))

        np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "y_pred.npy"), best_y_pred)
        np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "y_true.npy"), best_y_true)

        for k, v in losses.items():
            np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "losses_{}.npy".format(k)),
                    np.array(v))
        for k, v in r2s.items():
            np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "rsq_{}.npy".format(k)),
                    np.array(v))

        save_model_path = os.path.join(home_dir, "models/", model_name, epoch_prefix, "saved_model.model")
        torch.save(model.state_dict(), save_model_path)
    """