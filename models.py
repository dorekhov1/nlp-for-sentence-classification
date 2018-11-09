import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        ######

        # 4.1 YOUR CODE HERE

        ######

    def forward(self, x, lengths=None):
        pass
        ######

        # 4.1 YOUR CODE HERE

        ######


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        ######

        # 4.2 YOUR CODE HERE

        ######

    def forward(self, x, lengths=None):
        pass
        ######

        # 4.2 YOUR CODE HERE

        ######

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######

        # 4.3 YOUR CODE HERE

        ######


    def forward(self, x, lengths=None):
        pass
        ######

        # 4.3 YOUR CODE HERE

        ######
