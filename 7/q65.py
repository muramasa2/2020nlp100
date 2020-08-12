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

semantic_analogy = [lst[-2:] for lst in dataset if not lst[0].startswith('gram')]
syntactic_analogy = [lst[-2:] for lst in dataset if lst[0].startswith('gram')]

acc = np.mean([true == pred for true, pred in semantic_analogy])
print('意味的アナロジー　正解率：', acc)

acc = np.mean([true == pred for true, pred in syntactic_analogy])
print('文法的アナロジー　正解率：', acc)
