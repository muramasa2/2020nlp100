import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


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


countries = {
    country
    for lst in dataset
    for country in [lst[2], lst[4]]
    if lst[0] in {'capital-common-countries', 'capital-world'}
} | {
    country
    for lst in dataset
    for country in [lst[1], lst[3]]
    if lst[0] in {'currency', 'gram6-nationality-adjective'}
}

countries = list(countries)
len(countries)

country_vectors = [model[country] for country in countries]

plt.figure(figsize=(16, 9), dpi=200)
Z = linkage(country_vectors, method='ward')
dendrogram(Z, labels = countries)
plt.show()
