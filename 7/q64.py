import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors

similarity_path = './data/GoogleNews-vectors-negative300.bin'
dataset_path = './data/questions-words.txt'

model = KeyedVectors.load_word2vec_format(similarity_path, binary=True)

with open(dataset_path, 'r') as f:
    lines = f.read().splitlines()

dataset = []
category = None

for line in lines:
    if line.startswith(':'):
        category = line[2:]
    else:
        lst = [category] + line.split(' ')
        dataset.append(lst)

for idx, lst in enumerate(tqdm(dataset)):
    pred, prob = model.most_similar(positive=lst[2:4], negative=lst[1:2], topn=1)[0]
    dataset[idx].append(pred)

pd.DataFrame(dataset[:10])
