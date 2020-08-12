import pandas as pd
from glob import glob
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))  # str,maketrans('abcd', 'efgh') or str,maketrans({'a':'e', 'b':'f', 'c':'g', 'd':'h'})で置換テーブルつくる
    text = text.translate(table)
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)

    return text

train_csv_path = './6/train.txt'
val_csv_path = './6/valid.txt'
test_csv_path = './6/test.txt'

train_df = pd.read_csv(train_csv_path, delimiter='\t')
val_df = pd.read_csv(val_csv_path, delimiter='\t')
test_df = pd.read_csv(test_csv_path, delimiter='\t')

train_df.head()

df = pd.concat([train_df, val_df, test_df])
df.reset_index(drop=True, inplace=True)

df.head()
df['TITLE'] = df['TITLE'].map(lambda x: preprocessing(x))

df.head()
train_valid = df[:len(train_df) + len(val_df)]
test = df[len(train_df) + len(val_df):]

train_vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))
test_vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))


train_valid_X  = train_vec_tfidf.fit_transform(train_valid['TITLE'])
test_X = test_vec_tfidf.fit_transform(test['TITLE'])

train_valid_X = pd.DataFrame(train_valid_X.toarray(), columns=train_vec_tfidf.get_feature_names())
test_X = pd.DataFrame(test_X.toarray(), columns=test_vec_tfidf.get_feature_names())

train_X = train_valid_X[:len(train_df)]
val_X = train_valid_X[len(train_df):]

train_X.head()

train_X.to_csv('./6/train.feature.txt', sep='\t', index=False)
val_X.to_csv('./6/valid.feature.txt', sep='\t', index=False)
test_X.to_csv('./6/test.feature.txt', sep='\t', index=False)
