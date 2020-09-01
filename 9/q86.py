import re
import torch
import string
import pickle
import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from collections import Counter
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence as pad


vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
train_X = pd.read_csv('./6/train.txt', sep='\t')['TITLE'][1:]
val_X = pd.read_csv('./6/valid.txt', sep='\t')['TITLE'][1:]
test_X = pd.read_csv('./6/test.txt', sep='\t')['TITLE'][1:]

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


class CNNDataset(Dataset):
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


class CNNClassifier(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.conv = nn.Conv1d(e_size, h_size, 3, padding=1)
        self.act = nn.ReLU()
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, 0, 0.1)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, batch):
        x = self.embed(batch['src'])
        x = self.dropout(x)
        x = self.conv(x.transpose(-1, -2))
        x = self.act(x)
        x = self.dropout(x)
        x.masked_fill_(batch['mask'].unsqueeze(-2) == 0, -1e4)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = self.out(x)

        return x


def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=8):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler(dataset, width, shuffle),
        collate_fn=dataset.collate,
        num_workers=num_workers,
    )


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

cnn_train_dataset = CNNDataset(cnn_train_s, train_y)
cnn_val_dataset = CNNDataset(cnn_val_s, val_y)
cnn_test_dataset = CNNDataset(cnn_test_s, test_y)

device = torch.device('cpu')
model = CNNClassifier(len(vocab_dict), 300, 128, 4)
loader = gen_loader(cnn_test_dataset, 4000, num_workers=0)
iter(loader).next()
model(iter(loader).next()).argmax(dim=-1)
