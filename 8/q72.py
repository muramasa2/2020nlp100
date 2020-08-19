import pandas as pd
from gensim.models import KeyedVectors
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch


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
pred_y = torch.softmax(x, dim=-1)
criterion = nn.CrossEntropyLoss()
loss = criterion(pred_y, train_y[:4])
model.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配 :', model.fc.weight.grad)



x = model(train_X[:1])
pred_y = torch.softmax(x, dim=-1)
criterion = nn.CrossEntropyLoss()
loss = criterion(pred_y, train_y[:1])
model.zero_grad()
loss.backward()
print('損失 :', loss.item())
print('勾配 :', model.fc.weight.grad)
