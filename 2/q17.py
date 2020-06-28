import sys

names = (line.split('\t')[0] for line in sys.stdin)

print('\n'.join(names))

# cut -f1 popular-names.txt | sort | uniq
