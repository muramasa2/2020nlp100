from transformers import *
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
import numpy as np
from gensim.models import KeyedVectors


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
    def __iter__(self):
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
            this_len = self.dataset.lengths[index]
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


class BertDataset(Dataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        src = [torch.cat([x['src'], torch.zeros(max_seq_len - x['lengths'], dtype=torch.long)], dim=-1) for x in xs]
        src = torch.stack(src)
        mask = [[1] * int(x['lengths']) + [0] * int(max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'src':src,
            'trg':torch.tensor([x['trg'] for x in xs]),
            'mask':mask,
        }


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-cased', num_labels=4)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-cased', config=config)

    def forward(self, batch):
        x = self.bert(batch['src'], attention_mask=batch['mask'])
        return x[0]


class Task:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, model, batch):
        model.zero_grad()
        loss = self.criterion(model(batch), batch['trg'])
        loss.backward()
        return loss.item()

    def valid_step(self, model, batch):
        with torch.no_grad():
            loss = self.criterion(model(batch), batch['trg'])
        return loss.item()

class Trainer:
    def __init__(self, model, loaders, task, optimizer, max_iter, device = None):
        self.model = model
        self.model.to(device)
        self.train_loader, self.valid_loader = loaders
        self.task = task
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.device = device

    def send(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch

    def train_epoch(self):
        self.model.train()
        acc = 0
        for n, batch in enumerate(self.train_loader):
            batch = self.send(batch)
            acc += self.task.train_step(self.model, batch)
            self.optimizer.step()
        return acc / n

    def valid_epoch(self):
        self.model.eval()
        acc = 0
        for n, batch in enumerate(self.valid_loader):
            batch = self.send(batch)
            acc += self.task.valid_step(self.model, batch)
        return acc / n

    def train(self):
        for epoch in range(self.max_iter):
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()
            print('epoch {}, train_loss:{:.5f}, valid_loss:{:.5f}'.format(epoch, train_loss, valid_loss))


class Predictor:
    def __init__(self, model, loader, device=None):
        self.model = model
        self.loader = loader
        self.device = device

    def send(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch

    def infer(self, batch):
        self.model.eval()
        batch = self.send(batch)
        return self.model(batch).argmax(dim=-1).item()

    def predict(self):
        lst = []
        for batch in self.loader:
            lst.append(self.infer(batch))
        return lst


def read_for_bert(filename):
    with open(filename, encoding='utf8') as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset[1:]]
    dataset_t = [categories.index(line[4]) for line in dataset]
    dataset_X = [torch.tensor(tokenizer.encode(line[1]), dtype=torch.long) for line in dataset]
    return dataset_X, torch.tensor(dataset_t, dtype=torch.long)

def accuracy(true, pred):
    return np.mean([t==p for t,p in zip(true, pred)])


def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=8):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler = sampler(dataset, width, shuffle),
        collate_fn = dataset.collate,
        num_workers = num_workers,
    )


def gen_descending_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler = DescendingSampler, shuffle = False, num_workers = num_workers)


def gen_maxtokens_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler = MaxTokensSampler, shuffle = True, num_workers = num_workers)


with open('data/train_y.pickle', 'rb') as f:
    train_y = pickle.load(f)

with open('data/val_y.pickle', 'rb') as f:
    val_y = pickle.load(f)

with open('data/test_y.pickle', 'rb') as f:
    test_y = pickle.load(f)


train_y = torch.tensor(train_y)
val_y = torch.tensor(val_y)
test_y = torch.tensor(test_y)

categories = ['b', 't', 'e', 'm']
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

bert_train_X, bert_train_t = read_for_bert('data/train.txt')
bert_val_X, bert_val_t = read_for_bert('data/valid.txt')
bert_test_X, bert_test_t = read_for_bert('data/test.txt')

bert_train_dataset = BertDataset(bert_train_X, bert_train_t)
bert_val_dataset = BertDataset(bert_val_X, bert_val_t)
bert_test_dataset = BertDataset(bert_test_X, bert_test_t)

model = BertClassifier()
loaders = (
    gen_maxtokens_loader(bert_train_dataset, 1000),
    gen_descending_loader(bert_val_dataset, 32),
)
task = Task()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
device = torch.device('cpu')
trainer = Trainer(model, loaders, task, optimizer, max_iter=5, device)
trainer.train()

predictor = Predictor(model, gen_loader(bert_train_dataset, 1, num_workers=0), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_y, pred))

predictor = Predictor(model, gen_loader(bert_test_dataset, 1, num_workers=0), device)
pred = predictor.predict()
print('テストデータでの正解率 :', accuracy(test_y, pred))
