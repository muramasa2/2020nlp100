import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
import string
import re
from collections import Counter
import pickle


def preprocessing(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))  # str,maketrans('abcd', 'efgh') or str,maketrans({'a':'e', 'b':'f', 'c':'g', 'd':'h'})で置換テーブルつくる
    # string.punctuation = 英数字以外のアスキー文字のこと ex) !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(table)
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)

    return text

train_X = pd.read_csv('./6/train.txt', sep='\t')
val_X = pd.read_csv('./6/valid.txt', sep='\t')
test_X = pd.read_csv('./6/test.txt', sep='\t')

train_X.head()

train_y = train_X['CATEGORY']
counter = Counter([word for sent in train_X['TITLE'] for word in preprocessing(sent).split()])
vocab = [token for token, freq in counter.most_common() if freq > 1]
len(vocab)

[word for sent in train_X['TITLE'] for word in preprocessing(sent).split()]

vocab_list = ['[UNK]'] + vocab
vocab_dict = {x:n for n, x in enumerate(vocab_list)}

def sent2id(sent):
    ids = [vocab_dict[word if word in vocab else '[UNK]'] for word in preprocessing(sent).split()]

    return torch.tensor(ids, dtype=torch.long)


sent2id(train_X['TITLE'][0])


def dataset2ids(dataset):
    return [sent2id(x) for x in dataset]

train_s = dataset2ids(train_X['TITLE'])
val_s = dataset2ids(val_X['TITLE'])
test_s = dataset2ids(test_X['TITLE'])

with open('data/train_s.yaml', 'wb') as f:
    pickle.dump(train_s ,f)

with open('data/val_s.yaml', 'wb') as f:
    pickle.dump(val_s ,f)

with open('data/test_s.yaml', 'wb') as f:
    pickle.dump(test_s ,f)

with open('data/vocab_dict.pickle', 'wb') as f:
    pickle.dump(vocab_dict ,f)
