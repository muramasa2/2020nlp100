import sys

with open('col1.txt', 'w') as f1, open('col2.txt', 'w') as f2:
    for line in sys.stdin:
        cols = line.strip('\n').split('\t')
        print(cols[0], file=f1)
        print(cols[1], file=f2)

# cut -f1 popular-names.txt > col1a.txt && -f2 popular-names.txt > col2a.txt
