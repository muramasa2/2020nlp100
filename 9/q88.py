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
from tqdm import tqdm


vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
train_X = pd.read_csv('./6/train.txt', sep='\t')['TITLE']
val_X = pd.read_csv('./6/valid.txt', sep='\t')['TITLE']
test_X = pd.read_csv('./6/test.txt', sep='\t')['TITLE']


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


counter = Counter([
    x
    for sent in train_X
    for x in sent
])

vocab_in_train = [
    token
    for token, freq in counter.most_common()
    if freq > 1
]

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
    def __init__(self, dataset, width, shuffle = False):
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


class BiLSTMClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, num_layers=2, bidirectional=True)
        self.out = nn.Linear(h_size*2, c_size)
        self.dropout = nn.Dropout(dropout)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)
        nn.init.uniform_(self.out.weight, -0.1, 0.1)

    def forward(self, batch, h=None):
        x = self.embed(batch['src'])
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        h = h[-2:]
        h = h.transpose(0,1)
        h = h.contiguous().view(-1, h.size(1)*h.size(2))
        h = self.out(h)
        return h


class Task:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, model, batch):
        model.zero_grad()
        loss = self.criterion(model(batch), batch['trg'])
        loss.backward()
        return loss.item()

    def val_step(self, model, batch):
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
            acc += self.task.val_step(self.model, batch)
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


class BiLSTMCNNDataset(Dataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        mask = [[1] * int(x['lengths']) + [0] * int(max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'src':pad([x['src'] for x in xs]),
            'trg':torch.stack([x['trg'] for x in xs], dim=-1),
            'mask':mask,
            'lengths':torch.stack([x['lengths'] for x in xs], dim=-1)
        }


class BiLSTMCNNClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, bidirectional = True)
        self.conv = nn.Conv1d(h_size* 2, h_size, 3, padding=1)
        self.act = nn.ReLU()
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, batch, h=None):
        x = self.embed(batch['src'])
        x = self.dropout(x)
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        x, _ = unpack(x)
        x = self.dropout(x)
        x = self.conv(x.permute(1, 2, 0))
        x = self.act(x)
        x = self.dropout(x)
        x.masked_fill_(batch['mask'].unsqueeze(-2) == 0, -1)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = self.out(x)
        return x


def init_cnn_embed(embed):
    for i, token in enumerate(cnn_vocab_list):
        if token in vectors:
            embed.weight.data[i] = torch.from_numpy(vectors[token])
    return embed


def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])


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


def preprocessing(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))  # str,maketrans('abcd', 'efgh') or str,maketrans({'a':'e', 'b':'f', 'c':'g', 'd':'h'})で置換テーブルつくる
    # string.punctuation = 英数字以外のアスキー文字のこと ex) !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(table)
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)
    if text in string.ascii_letters or string.digits:
        return text
    else:
        return str(0)


def cnn_sent_to_ids(sent):
    return torch.tensor([cnn_vocab_dict[x if x in cnn_vocab_dict else '[UNK]'] for x in sent], dtype=torch.long)


def cnn_dataset_to_ids(dataset):
    return [cnn_sent_to_ids(preprocessing(x)) for x in dataset]


cnn_vocab_list = ['[PAD]', '[UNK]'] + vocab_in_train
cnn_vocab_dict = {x:n for n, x in enumerate(cnn_vocab_list)}

cnn_train_s = cnn_dataset_to_ids(train_X)
cnn_val_s = cnn_dataset_to_ids(val_X)
cnn_test_s = cnn_dataset_to_ids(test_X)

device = torch.device('cpu')
rnncnn_train_dataset = BiLSTMCNNDataset(cnn_train_s, train_y)
rnncnn_val_dataset = BiLSTMCNNDataset(cnn_val_s, val_y)
rnncnn_test_dataset = BiLSTMCNNDataset(cnn_test_s, test_y)

loaders = (
    gen_maxtokens_loader(rnncnn_train_dataset, 4000),
    gen_descending_loader(rnncnn_val_dataset, 32),
)
task = Task()
for h in tqdm([32, 64, 128, 256, 512]):
    model = BiLSTMCNNClassifier(len(cnn_vocab_dict), 300, h, 4)  # v_size, e_size, h_size, c_size
    init_cnn_embed(model.embed)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)
    trainer = Trainer(model, loaders, task, optimizer, 10, device)
    trainer.train()
    predictor = Predictor(model, gen_loader(rnncnn_test_dataset, 1, num_workers=0), device)
    pred = predictor.predict()
    print('評価データでの正解率 :', accuracy(test_y, pred))
