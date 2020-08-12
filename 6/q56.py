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

def score_lg(lg, X):
    return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

train_X.head()
test_X.head()
train_pred = score_lg(lg, train_X)
missing_cols = set(train_X.columns) - set(test_X.columns)

for col in missing_cols:
    test_X[col] = 0
test_X.head()

test_pred = score_lg(lg, test_X)
print(train_pred)
print(test_pred)

recall = recall_score(test_y, test_pred[1], average=None)
precision = precision_score(test_y, test_pred[1], average=None)
f1 = f1_score(test_y, test_pred[1], average=None)

micro_recall = recall_score(test_y, test_pred[1], average='micro')
micro_precision = precision_score(test_y, test_pred[1], average='micro')
micro_f1 = f1_score(test_y, test_pred[1], average='micro')

macro_recall = recall_score(test_y, test_pred[1], average='macro')
macro_precision = precision_score(test_y, test_pred[1], average='macro')
macro_f1 = f1_score(test_y, test_pred[1], average='macro')


prec = np.hstack([precision, micro_precision, macro_precision])
rec = np.hstack([recall, micro_recall, macro_recall])
f = np.hstack([f1, micro_f1, macro_f1])
np.vstack([prec, rec, f])
df = pd.DataFrame(np.vstack([prec, rec, f]).T, columns=['precision', 'recall', 'f1'], index=['b', 'e', 't', 'm', 'micro', 'macro'])
df
