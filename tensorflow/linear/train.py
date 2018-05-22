"""Train the model"""

import argparse
import logging
import os
import sys

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_text
from model.model_fn import model_fn

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(69)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    """OMIT:
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path)
    """
    #OMIT: num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_dir is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Get paths for vocabularies and dataset
    #OMIT: path_words = os.path.join(args.data_dir, 'words.txt')
    #OMIT: path_tags = os.path.join(args.data_dir, 'tags.txt')
    path_train_features = os.path.join(args.data_dir, 'train/features.txt')
    path_train_homeruns = os.path.join(args.data_dir, 'train/homeruns.txt')
    path_eval_features = os.path.join(args.data_dir, 'dev/features.txt')
    path_eval_homeruns = os.path.join(args.data_dir, 'dev/homeruns.txt')

    # Load Vocabularies
    #OMIT: words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=num_oov_buckets)
    #OMIT: tags = tf.contrib.lookup.index_table_from_file(path_tags)

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    train_features = load_dataset_from_text(path_train_features, "features")
    train_homeruns = load_dataset_from_text(path_train_homeruns, "homeruns")
    eval_features = load_dataset_from_text(path_eval_features, "features")
    eval_homeruns = load_dataset_from_text(path_eval_homeruns, "homeruns")

    # Specify other parameters for the dataset and the model
    #FIX to not hard code: params.eval_size = params.dev_size
    #FIX to not hard code: params.buffer_size = params.train_size # buffer size for shuffling
    params.eval_size = 3357
    params.buffer_size = 15663 # buffer size for shuffling
    params.train_size = 15663
    #OMIT: params.id_pad_word = words.lookup(tf.constant(params.pad_word))
    #OMIT: params.id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_features, train_homeruns, params)
    eval_inputs = input_fn('eval', eval_features, eval_homeruns, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)