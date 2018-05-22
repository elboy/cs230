"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
import random

game_index = 5
at_bats_index = 6
double_index = 11
triple_index = 11
homerun_index = 11
#feature_indices = np.array([5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21])

def create_feature_vector(row):

    feature_vector = row[game_index:]
    del feature_vector[homerun_index - game_index]
    del feature_vector[triple_index - game_index]
    del feature_vector[double_index - game_index]
    return feature_vector

def load_dataset(path_csv, at_bats_threshold=0, filter_null=True):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    
    use_python3 = sys.version_info[0] >= 3
    with (open(path_csv, encoding="windows-1252") if use_python3 else open(path_csv)) as f:
        csv_file = csv.reader(f, delimiter=',')
        dataset = []
        words, tags = [], []

        # Each line of the csv corresponds to one word
        for idx, row in enumerate(csv_file):
            if idx == 0: continue
                
            at_bats = row[at_bats_index]
            if int(at_bats) < at_bats_threshold: continue
                
            homeruns = row[homerun_index]
            feature_vector = create_feature_vector(row)
            
            if filter_null and '' in feature_vector: continue
                
            dataset.append((feature_vector, homeruns))

            """
            except UnicodeDecodeError as e:
                print("An exception was raised, skipping a word: {}".format(e))
                pass
            """

    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'features.txt'), 'w') as feature_file:
        with open(os.path.join(save_dir, 'homeruns.txt'), 'w') as homerun_file:
            for features, homeruns in dataset:
                feature_file.write("{}\n".format(" ".join(features)))
                homerun_file.write("{}\n".format(homeruns))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/kaggle/Batting.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg
    
    at_bats_threshold = 100

    # Load the dataset into memory
    print("Loading Kaggle dataset into memory...")
    dataset = load_dataset(path_dataset, at_bats_threshold, True)
    print("Parsed {} player seasons.".format(len(dataset)))
    print("- done.")
    
    random.seed(10)
    random.shuffle(dataset)
    
    # Split the dataset into train, dev and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7*len(dataset))]
    dev_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/kaggle/train')
    save_dataset(dev_dataset, 'data/kaggle/dev')
    save_dataset(test_dataset, 'data/kaggle/test')