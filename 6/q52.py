import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

train_csv_path = './6/train.txt'
val_csv_path = './6/valid.txt'
test_csv_path = './6/test.txt'


train_df = pd.read_csv(train_csv_path, delimiter='\t')
val_df = pd.read_csv(val_csv_path, delimiter='\t')
test_df = pd.read_csv(test_csv_path, delimiter='\t')

train_df.head()

train_y = train_df['CATEGORY']
# train_X = train_df.drop(3, axis=1)
#
#
# oh_enc_train_X = pd.get_dummies(train_X)
#
# clf = LogisticRegression()
# clf.fit(oh_enc_train_X, train_y)
# test_y = test_df.iloc[:, 3]
# test_X = test_df.drop(3, axis=1)
#
# # Get missing columns in the training test
#
# oh_enc_test_X = pd.get_dummies(test_X)
# test_X.head()
#
# missing_cols = set(oh_enc_train_X.columns) - set(oh_enc_test_X.columns)
# # Add a missing column in test set with default value equal to 0
# for col in missing_cols:
#     oh_enc_test_X[col] = 0
#
# # Ensure the order of column in the test set is in the same order than in train set
# oh_enc_test_X = oh_enc_test_X[oh_enc_train_X.columns]
# test_y.unique()
# oh_enc_test_X.head()
# clf.predict(oh_enc_test_X).shape
# pred_y = clf.predict(oh_enc_test_X)
# np.unique(pred_y)
# clf.predict_proba(oh_enc_test_X)[0]



train_X = pd.read_csv('./6/train.feature.txt', sep='\t')
val_X = pd.read_csv('./6/valid.feature.txt', sep='\t')
test_X = pd.read_csv('./6/test.feature.txt', sep='\t')

lg = LogisticRegression(random_state=1, max_iter=10000)
lg.fit(train_X, train_y)
