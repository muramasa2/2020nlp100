from scipy.stats import spearmanr
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


similarity_path = './data/GoogleNews-vectors-negative300.bin'
dataset_path = './data/questions-words.txt'

with open('./data/wordsim353/combined.csv') as f:
    data = f.read()

data = data.splitlines()
data = data[1:]
data = [line.split(',') for line in data]

model = KeyedVectors.load_word2vec_format(similarity_path, binary=True)

for idx, lst in enumerate(data):
    sim = model.similarity(lst[0], lst[1])
    data[idx].append(sim)

pd.DataFrame(data[:10], columns = ['単語1', '単語2', '類似度(人力)', '類似度(ベクトル)'])

def rank(x):
    args = np.argsort(-np.array(x))
    rank = np.empty_like(args)
    rank[args] = np.arange(len(x))
    return rank

human = [float(lst[2]) for lst in data]
w2v = [lst[3] for lst in data]

human_rank = rank(human)
w2v_rank = rank(w2v)
rho, p_value = spearmanr(human_rank, w2v_rank)

plt.scatter(human_rank, w2v_rank)
plt.show()
