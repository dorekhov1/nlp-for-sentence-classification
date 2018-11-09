import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os

import pandas as pd

# from models import *


def main(args):
    ######

    # 3.2 Processing of the data

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    TEXT = data.Field(sequential=True, tokenize='spacy', lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(
        path='data/',
        train='train.tsv',
        validation='validation.tsv',
        test='test.tsv',
        format='tsv',
        skip_header=True,
        fields=[('Text', TEXT), ('Label', LABEL)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        datasets=(train, val, test),
        batch_size=(64, 64, 128),
        sort_key=lambda x: len(x.Text),
        sort_within_batch=True,
        repeat=False,
        device=device
    )

    TEXT.build_vocab(train)
    vocab = TEXT.vocab
    vocab.load_vectors(vectors="glove.6B.100d")

    ######

    ######

    # 5 Training and Evaluation

    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
