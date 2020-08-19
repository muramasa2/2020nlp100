import pandas as pd
from gensim.models import KeyedVectors
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch


word2vec_path = './data/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

train_path = './data/train.txt'
valid_path = './data/valid.txt'
test_path = './data/test.txt'

train_df = pd.read_csv(train_path, delimiter='\t')
val_df = pd.read_csv(valid_path, delimiter='\t')
test_df = pd.read_csv(test_path, delimiter='\t')

train_df.head()

train_df.shape
val_df.shape
test_df.shape
train_X = train_df['TITLE']
val_X = val_df['TITLE']
test_X = test_df['TITLE']
def sent2vec(df):
    word2vec_dim = 300
    X = np.zeros((df.shape[0], 300))
    sent_list = df['TITLE']
    for sent_idx, sent in tqdm(enumerate(sent_list)):
        list = [model[word] for word in sent.split() if word in model]
        X[sent_idx] = sum(list) / len(sent.split())

    return X


train_X = sent2vec(train_df)
val_X = sent2vec(val_df)
test_X = sent2vec(test_df)

train_df['CATEGORY'].unique()
trans_dict = {'b':0, 't':1, 'e':2, 'm':3}

train_y = [trans_dict[x] for x in list(train_df['CATEGORY'])]
val_y = [trans_dict[x] for x in list(val_df['CATEGORY'])]
test_y = [trans_dict[x] for x in list(val_df['CATEGORY'])]


with open('data/train_X.pickle', 'wb') as f:
    pickle.dump(train_X, f)
with open('data/train_y.pickle', 'wb') as f:
    pickle.dump(train_y, f)

with open('data/val_X.pickle', 'wb') as f:
    pickle.dump(val_X, f)
with open('data/val_y.pickle', 'wb') as f:
    pickle.dump(val_y, f)

with open('data/test_X.pickle', 'wb') as f:
    pickle.dump(test_X, f)
with open('data/test_y.pickle', 'wb') as f:
    pickle.dump(test_y, f)


with open('data/train_X.pickle', 'rb') as f:
    train_X = pickle.load(f)
with open('data/train_y.pickle', 'rb') as f:
    train_y = pickle.load(f)

with open('data/val_X.pickle', 'rb') as f:
    val_X = pickle.load(f)
with open('data/val_y.pickle', 'rb') as f:
    val_y = pickle.load(f)

with open('data/test_X.pickle', 'rb') as f:
    test_X = pickle.load(f)
with open('data/test_y.pickle', 'rb') as f:
    test_y = pickle.load(f)

train_X = torch.tensor(train_X).float()
train_y = torch.tensor(train_y).long()
val_X = torch.tensor(val_X).float()
val_y = torch.tensor(val_y).long()
test_X = torch.tensor(test_X).float()
test_y = torch.tensor(test_y).long()


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias = False)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

model = Perceptron(300, 4)
x = model(train_X[:4])
x = torch.softmax(x, dim=-1)

x
train_y[:4]
