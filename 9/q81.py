import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
import string
import re
from collections import Counter
import pickle
import torch.utils.data
import random as rd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


with open('data/train_s.yaml', 'rb') as f:
    train_s = pickle.load(f)

with open('data/val_s.yaml', 'rb') as f:
    val_s = pickle.load(f)

with open('data/test_s.yaml', 'rb') as f:
    test_s = pickle.load(f)

with open('data/vocab_dict.pickle', 'rb') as f:
    vocab_dict = pickle.load(f)

with open('data/train_y.pickle', 'rb') as f:
    train_y = pickle.load(f)

with open('data/val_y.pickle', 'rb') as f:
    val_y = pickle.load(f)

with open('data/test_y.pickle', 'rb') as f:
    test_y = pickle.load(f)


train_y = torch.tensor(train_y)
val_y = torch.tensor(val_y)
test_y = torch.tensor(test_y)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.lengths = torch.tensor([len(x) for x in source])
        self.size = len(source)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            'src':self.source[index],
            'trg':self.target[index],
            'lengths':self.lengths[index],
        }

    def collate(self, xs):
        return {
        'src':pad([x['src'] for x in xs]),
        'trg':torch.stack([x['trg'] for x in xs], dim=-1),
        'lengths':torch.stack([x['lengths'] for x in xs], dim=-1)
        }

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, width, shuffle=False):
        self.dataset = dataset
        self.width = width
        self.shuffle = shuffle
        if not shuffle:
            self.indices = torch.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))
        index = 0
        while index < len(self.dataset):
            yield self.indices[index : index + self.width]
            index += self.width

class DescendingSampler(Sampler):
    def __init__(self, dataset, width, shuffle = False):
        assert not shuffle
        super().__init__(dataset, width, shuffle)
        self.indices = self.indices[self.dataset.lengths[self.indices].argsort(descending=True)]


class MaxTokensSampler(Sampler):
    def __init__(self):
        self.indices = torch.randperm(len(self.dataset))
        self.indices = self.indices[self.dataset.lengths[self.indices].argsort(descending=True)]
        for batch in self.generate_batches():
            yield batch

    def generate_batches(self):
        batches = []
        batch = []
        acc = 0
        max_len = 0
        for index in self.indices:
            acc += 1
            this_len = self.dataset.length[index]
            max_len = max(max_len, this_len)
            if acc * max_len > self.width:
                batches.append(batch)
                batch = [index]
                acc = 1
                max_len = this_len
            else:
                batch.append(index)
        if batch != []:
            batches.append(batch)
        rd.shuffle(batches)
        return batches

class LSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, num_layers=1)
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            param.data.uniform_(-0.1, 0.1)
        self.out.weight.data.uniform_(-0.1, 0.1)

    def forward(self, batch, h=None):
        x = self.embed(batch['src'])
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        h = self.out(h)
        return h.squeeze(0)


def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=8):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler(dataset, width, shuffle),
        collate_fn=dataset.collate,
        num_workers=num_workers,
    )


def gen_descending_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler=DescendingSampler,
                      shuffle=False, num_workers=num_workers)


def gen_maxtokens_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler=MaxTokensSampler,
                      shuffle=True, num_workers=num_workers)


train_dataset = Dataset(train_s, train_y)
val_dataset = Dataset(val_s, val_y)
test_dataset = Dataset(test_s, test_y)

model = LSTMClassifier(len(vocab_dict), 300, 50, 4)
loader = gen_loader(test_dataset, 10, DescendingSampler, False, num_workers=0)
iter(loader).next()
model(iter(loader).next()).argmax(dim=-1)
