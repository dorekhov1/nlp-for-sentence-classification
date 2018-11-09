"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data():
    return pd.read_csv("data/data.tsv", sep="\t")


def split_data(ones, zeroes, s):

    train_ones, test_ones, train_zeroes, test_zeroes = train_test_split(ones, zeroes, test_size=0.20, random_state=s)
    train_ones, val_ones, train_zeroes, val_zeroes = train_test_split(train_ones, train_zeroes, test_size=0.20, random_state=s)

    return pd.concat([train_ones, train_zeroes]), pd.concat([test_ones, test_zeroes]), pd.concat([val_ones, val_zeroes])


def save_data(train_data, test_data, val_data):
    train_data.to_csv("data/train.tsv", sep="\t", index=False)
    test_data.to_csv("data/test.tsv", sep="\t", index=False)
    val_data.to_csv("data/validation.tsv", sep="\t", index=False)


def print_balancing(train_data, test_data, val_data):
    print("Number of ones in train: ", train_data[train_data.label == 1].shape[0])
    print("Number of zeroes in train: ", train_data[train_data.label == 0].shape[0])

    print("Number of ones in test: ", test_data[test_data.label == 1].shape[0])
    print("Number of zeroes in test: ", test_data[test_data.label == 0].shape[0])

    print("Number of ones in validation: ", val_data[val_data.label == 1].shape[0])
    print("Number of zeroes in validation: ", val_data[val_data.label == 0].shape[0])


if __name__ == "__main__":
    data = load_data()
    train, test, val = split_data(data[data.label == 1], data[data.label == 0], 0)
    print_balancing(train, test, val)
    save_data(shuffle(train), shuffle(test), shuffle(val))
