import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train_csv_path = './6/train.txt'
val_csv_path = './6/valid.txt'
test_csv_path = './6/test.txt'

train_df = pd.read_csv(train_csv_path, delimiter='\t')
val_df = pd.read_csv(val_csv_path, delimiter='\t')
test_df = pd.read_csv(test_csv_path, delimiter='\t')

train_df.head()

train_y = train_df['CATEGORY']
val_y = val_df['CATEGORY']
test_y = test_df['CATEGORY']

train_X = pd.read_csv('./6/train.feature.txt', sep='\t')
val_X = pd.read_csv('./6/valid.feature.txt', sep='\t')
test_X = pd.read_csv('./6/test.feature.txt', sep='\t')

lg = LogisticRegression(random_state=1, max_iter=10000)
lg.fit(train_X, train_y)
train_X.shape
lg.predict_proba(train_X).shape
features = train_X.columns.values
index = [i for i in range(1, 11)]
lg.classes_.shape
lg.coef_.shape

for c, coef in zip(lg.classes_, lg.coef_):
    print(f'【カテゴリ】{c}')
    best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['重要度上位'], index=index).T
    worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T

    display(pd.concat([best10, worst10], axis=0))
    print('\n')
