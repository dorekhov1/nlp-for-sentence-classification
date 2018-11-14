import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.emb = nn.Embedding(len(vocab), embedding_dim)
        self.emb.from_pretrained(vocab.vectors)
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.emb(x).permute(1, 0, 2)
        x = self.fc1(torch.mean(x, 1)).squeeze()
        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(len(vocab), embedding_dim)
        self.emb.from_pretrained(vocab.vectors)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.emb(x)
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        _, h = self.rnn(x)
        # x = nn.utils.rnn.pad_packed_sequence(x, total_length=x.size(0))
        x = self.linear(h)

        return x.squeeze()


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(len(vocab), embedding_dim)
        self.emb.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.emb(x).permute(1, 0, 2).unsqueeze(1)
        x_1, _ = torch.max(self.conv1(x), 2)
        x_2, _ = torch.max(self.conv2(x), 2)
        x = torch.cat([x_1, x_2], 1).squeeze()
        x = self.linear(x).squeeze()
        return x
