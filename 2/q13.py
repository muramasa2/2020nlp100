import pandas as pd

col1 = pd.read_table('2/col1.txt', header=None)
col2 = pd.read_table('2/col2.txt', header=None)

new_data = pd.concat([col1, col2], axis=1)
new_data.to_csv('2/concat.txt', header=None, index=None, sep='\t')

############
# 模範解答 #
############

with open('2/col1.txt') as f1, \
    open('2/col2.txt') as f2:
    for col1, col2 in zip(f1, f2):
        col1 = col1.rstrip()
        col2 = col2.rstrip()
        print(f'{col1}\t{col2}')

# paste col1.txt col2.txt
