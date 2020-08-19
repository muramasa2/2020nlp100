import pandas as pd
from gensim.models import KeyedVectors
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data


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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, t):
        self.x = x
        self.t = t
        self.size = len(x)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            'x':self.x[index],
            't':self.t[index],
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

def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=0):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler = sampler(dataset, width, shuffle),
        num_workers = num_workers,
    )

class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias = False)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

train_dataset = Dataset(train_X, train_y)
val_dataset = Dataset(val_X, val_y)
test_dataset = Dataset(test_X, test_y)
loaders = (
    gen_loader(train_dataset, 1, shuffle = True),
    gen_loader(val_dataset, 1)
)

class Task:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, model, batch):
        model.zero_grad()
        loss = self.criterion(model(batch['x']), batch['t'])
        loss.backward()
        return loss.item()

    def valid_step(self, model, batch):
        with torch.no_grad():
            loss = self.criterion(model(batch['x']), batch['t'])
        return loss.item()

class Trainer:
    def __init__(self, model, loaders, task, optimizer, max_iter, device = None):
        self.model = model
        self.model.to(device)
        self.train_loader, self.valid_loader = loaders
        self.task = task
        self.max_iter = max_iter
        self.optimizer = optimizer
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

model = Perceptron(300, 4)
task = Task()
optimizer = optim.SGD(model.parameters(), 0.1)
trainer = Trainer(model, loaders, task, optimizer, 10)
trainer.train()
