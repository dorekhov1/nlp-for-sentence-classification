import torchtext
from torchtext import data
import spacy

import argparse
from models import *


def main(args):

    TEXT = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train = data.TabularDataset(
        "data/train.tsv",
        format="tsv",
        fields=[('Text', TEXT), ('Label', LABELS)]
    )

    TEXT.build_vocab(train)
    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    TEXT.vocab.load_vectors(glove)

    model_base = Baseline(args.emb_dim, TEXT.vocab)
    model_base = torch.load('model_baseline.pt')

    model_rnn = RNN(args.emb_dim, TEXT.vocab, args.rnn_hidden_dim)
    model_rnn = torch.load('model_rnn.pt')

    model_cnn = CNN(args.emb_dim, TEXT.vocab, args.num_filt, [2,4])
    model_cnn = torch.load('model_cnn.pt')

    models = [model_base, model_rnn, model_cnn]

    nlp = spacy.load('en')

    names = ["baseline", "rnn", "cnn"]

    while 1 == 1:
        print("Enter a sentence")
        x = input()

        tokenized = nlp(x)

        ints = []
        for token in tokenized:
            ints.append(TEXT.vocab.stoi[str(token)])

            # print(TEXT.vocab.stoi)

        x = torch.LongTensor(ints).unsqueeze(1)
        lengths = torch.IntTensor([len(ints)])

        print("\n")

        for i, model in enumerate(models):
            prediction = model.forward(x, lengths)

            if int(prediction.round()) == 1:
                cat = 'subjective'

            elif int(prediction.round()) == 0:
                cat = 'objective'

            print("Model {}: {} ({})".format(names[i], cat, str(prediction.detach().numpy())[:5]))

        print("\n")


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
    parser.add_argument('--log_freq', type=int, default=1)

    args = parser.parse_args()

    main(args)

