import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        ######

        # 4.1 YOUR CODE HERE
        self.emb = nn.Embedding.from_pretrained(vocab)

        ######

    def forward(self, x, lengths=None):

        ######

        # 4.1 YOUR CODE HERE

        x = self.emb(x)
        x = torch.mean(x)

        return x

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
