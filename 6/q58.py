import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


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

train_X.head()
val_X.head()
test_X.head()

train_X.shape
val_X.shape
test_X.shape

train_pred = score_lg(lg, train_X)
missing_cols = set(train_X.columns) - set(test_X.columns)

for col in missing_cols:
    test_X[col] = 0
test_X.head()

result = []
for C in tqdm(np.logspace(-5, 4, 10, base=10)):
    lg = LogisticRegression(random_state=1, max_iter=10000, C=C)
    lg.fit(train_X, train_y)

    train_pred = lg.predict(train_X)
    val_pred = lg.predict(val_X)
    test_pred = lg.predict(test_X)

    train_accuracy = accuracy_score(train_y.values, train_pred)
    val_accuracy = accuracy_score(val_y.values, val_pred)
    test_accuracy = accuracy_score(test_y.values, test_pred)

    result.append([C, train_accuracy, val_accuracy, test_accuracy])

result = np.array(result).T

plt.plot(result[0], result[1], label='train')
plt.plot(result[0], result[2], label='val')
plt.plot(result[0], result[3], label='test')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.xscale('log')
plt.xlabel('C')
plt.legend()
plt.show()
