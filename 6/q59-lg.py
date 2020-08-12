import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import optuna


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

def objective_lg(trial):
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    C = trial.suggest_uniform('C', 1e-4, 1e4)
    lg = LogisticRegression(random_state=1, max_iter=10000,
                            penalty='elasticnet', solver='saga',
                            l1_ratio=l1_ratio, C=C)
    lg.fit(train_X, train_y)
    val_pred = lg.predict(val_X)
    val_acc = accuracy_score(val_y, val_pred)

    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective_lg, timeout=3600)

print('Best trial:')
trial = study.best_trial
print(' Value: {:.3f}'.format(trial.value))
print(' Params: ')
for key, value in tqdm(trial.params.items()):
    print('  {}:{}'.format(key, value))

best_l1_ratio = trial.params['l1_ratio']
best_C = trial.params['l1_ratio']

lg = LogisticRegression(random_state=1, max_iter=10000,
                        penalty='elasticnet', solver='saga',
                        l1_ratio=best_l1_ratio, C=best_C)
lg.fit(train_X, train_y)

missing_cols = set(train_X.columns) - set(test_X.columns)
for col in missing_cols:
    test_X[col] = 0


train_pred = lg.predict(train_X)
val_pred = lg.predict(val_X)
test_pred = lg.predict(test_X)

train_accuracy = accuracy_score(train_y.values, train_pred)
val_accuracy = accuracy_score(val_y.values, val_pred)
test_accuracy = accuracy_score(test_y.values, test_pred)

print(f'正解率(train): {train_accuracy:.3f}')
print(f'正解率(val): {val_accuracy:.3f}')
print(f'正解率(test): {test_accuracy:.3f}')
