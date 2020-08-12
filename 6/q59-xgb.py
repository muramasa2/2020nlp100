import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb


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

params={'objective': 'multi:softmax',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.5,
        'min_child_weight': 1,
        'subsample': 0.9,
        'eta': 0.1,
        'max_depth': 5,
        'gamma': 0.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'num_round': 1000,
        'early_stopping_rounds': 50,
        'verbosity': 0
        }

category_dict = {'b': 0, 'e': 1, 't': 2, 'm': 3}
missing_cols = set(train_X.columns) - set(test_X.columns)

for col in missing_cols:
    test_X[col] = 0
test_X.head()
test_X = test_X.sort_index(axis=1, ascending=True)

train_y = [category_dict[c] for c in train_y]
val_y = [category_dict[c] for c in val_y]
test_y = [category_dict[c] for c in test_y]

dtrain = xgb.DMatrix(train_X, label=train_y)
dval = xgb.DMatrix(val_X, label=val_y)
dtest = xgb.DMatrix(test_X, label=test_y)

num_round = params.pop('num_round')
early_stopping_rounds = params.pop('early_stopping_rounds')
watchlist = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds)


train_pred = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
val_pred = model.predict(dval, ntree_limit=model.best_ntree_limit)
test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

train_accuracy = accuracy_score(train_y, train_pred)
val_accuracy = accuracy_score(val_y, val_pred)
test_accuracy = accuracy_score(test_y, test_pred)

print(f'正解率(train): {train_accuracy:.3f}')
print(f'正解率(val): {val_accuracy:.3f}')
print(f'正解率(test): {test_accuracy:.3f}')
