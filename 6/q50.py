import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


csv_path = './data/NewsAggregatorDataset/newsCorpora.csv'
df = pd.read_csv(csv_path, header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df.head()

extract_df = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
extract_df.head()

train_df, val_test_df = train_test_split(extract_df, test_size=0.2, shuffle=True, random_state=1, stratify=exstract_df['CATEGORY'])
val_df, test_df = train_test_split(val_test_df, test_size=0.5, shuffle=True, random_state=1, stratify=val_test_df['CATEGORY'])

print('raw_df length:', len(extract_df))
print('[train, val, test]df length:', list(map(len, [train_df, val_df, test_df])))

print(train_df['CATEGORY'].value_counts())
print(val_df['CATEGORY'].value_counts())
print(test_df['CATEGORY'].value_counts())

train_df.to_csv('./6/train.txt', index=False, sep='\t')
val_df.to_csv('./6/valid.txt', index=False, sep='\t')
test_df.to_csv('./6/test.txt', index=False, sep='\t')
