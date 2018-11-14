import torchtext
from torchtext import data

import argparse
import pandas as pd
from models import *
import copy


def set_up_data(bs):

    TEXT = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(
        path="data/",
        train="train.tsv",
        validation="validation.tsv",
        test="test.tsv",
        format="tsv",
        fields=[('Text', TEXT), ('Label', LABELS)],
        skip_header=True)

    # train_iter, val_iter, test_iter = data.BucketIterator.splits(
    #     datasets=(train, val, test),
    #     batch_sizes=(bs, bs, bs),
    #     sort_key=lambda x: len(x.Text),
    #     sort_within_batch=True,
    #     repeat=False)

    train_iter, val_iter, test_iter = data.Iterator.splits(
        datasets=(train, val, test),
        batch_sizes=(bs, bs, bs),
        sort_key=lambda x: len(x.Text),
        sort_within_batch=True,
        repeat=False)

    TEXT.build_vocab(train)
    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    TEXT.vocab.load_vectors(glove)

    return train_iter, val_iter, test_iter, TEXT.vocab


def evaluate(models, loss_f, iterator):
    err = [0, 0, 0]
    loss = [0, 0, 0]

    for batch in iterator:
        text, labels = batch.Text, batch.Label
        targets = labels.float().squeeze()

        for i, model in enumerate(models):
            predictions = model.forward(text[0], text[1])
            loss[i] += loss_f(predictions, targets).item()
            err[i] += int((predictions.round().int() != labels.int().squeeze()).sum())

    total_err = [e/len(iterator.dataset) for e in err]
    total_loss = [l/len(iterator) for l in loss]

    return total_err, total_loss


def training(models, opts, loss_f, train_iterator, val_iterator, args):

    acc = [10.0, 10.0, 10.0]
    err = [0, 0, 0]
    loss = [0, 0, 0]

    logs_b = {"Epoch": [], "Train Loss": [], "Train Error": [], "Val Loss": [], "Val Error": []}
    logs_rnn, logs_cnn = copy.deepcopy(logs_b), copy.deepcopy(logs_b)

    for epoch in range(args.epochs):
        for batch in train_iterator:
            text, labels = batch.Text, batch.Label
            targets = labels.float().squeeze()

            for i, (model, optimizer) in enumerate(zip(models, opts)):
                predictions = model.forward(text[0], text[1])

                optimizer.zero_grad()
                loss_val = loss_f(predictions, targets)
                loss_val.backward()
                optimizer.step()

                loss[i] += loss_f(predictions, targets).item()
                err[i] += int((predictions.round().int() != labels.int().squeeze()).sum())

        if epoch % args.eval_every == 0:
            v_err, v_loss = evaluate(models, loss_f, val_iterator)
            err = [e/len(train_iterator.dataset) for e in err]
            loss = [l/len(train_iterator) for l in loss]

            print("\nEpoch {}".format(epoch))
            logs_b, acc[0] = log("baseline", logs_b, epoch, err[0], loss[0], v_err[0], v_loss[0], acc[0], models[0])
            logs_rnn, acc[1] = log("rnn", logs_rnn, epoch, err[1], loss[1], v_err[1], v_loss[1], acc[1], models[1])
            logs_cnn, acc[2] = log("cnn", logs_cnn, epoch, err[2], loss[2], v_err[2], v_loss[2], acc[2], models[2])

            err = [0, 0, 0]
            loss = [0, 0, 0]

    return logs_b, logs_rnn, logs_cnn


def log(name, logs, epoch, err, loss, val_err, val_loss, acc, model):
    print("Model ", name)
    logs["Epoch"].append(epoch)
    logs["Train Loss"].append(loss)
    logs["Train Error"].append(err)
    logs["Val Error"].append(val_err)
    logs["Val Loss"].append(val_loss)
    print("Train Error: {}, Train Loss: {} | Val Error: {}, Val Loss: {}".format(err, loss, val_err, val_loss))

    if val_err < acc:
        acc = val_err
        print("Saving Model {} ... Best Error {}".format(name, val_err))
        torch.save(model, 'model_{}.pt'.format(name))

    return logs, acc


def main(args):

    train_iter, val_iter, test_iter, vocab = set_up_data(args.batch_size)

    model_base = Baseline(args.emb_dim, vocab)
    model_rnn = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)
    model_cnn = CNN(args.emb_dim, vocab, args.num_filt, [2, 4])

    models = [model_base, model_rnn, model_cnn]

    optimizer_base = torch.optim.Adam(model_base.parameters(), lr=args.lr)
    optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=args.lr)
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=args.lr)

    optimizers = [optimizer_base, optimizer_rnn, optimizer_cnn]

    loss = torch.nn.BCELoss()

    logs_base, logs_rnn, logs_cnn = training(models, optimizers, loss, train_iter, val_iter, args)
    test_err, test_loss = evaluate(models, loss, test_iter)

    print("\nBaseline Test: Error: {}, Loss: {}".format(test_err[0], test_loss[0]))
    print("RNN Test: Error: {}, Loss: {}".format(test_err[1], test_loss[1]))
    print("CNN Test: Error: {}, Loss: {}".format(test_err[2], test_loss[2]))

    pd.DataFrame(logs_base).to_csv("model_{}.csv".format("baseline"), index=False)
    pd.DataFrame(logs_rnn).to_csv("model_{}.csv".format("rnn"), index=False)
    pd.DataFrame(logs_cnn).to_csv("model_{}.csv".format("cnn"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=1)

    _args = parser.parse_args()

    main(_args)
