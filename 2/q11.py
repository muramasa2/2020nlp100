import sys

for line in sys.stdin:
    print(line.replace('\t', ' '))

# tr '\t' ' ' < popular-names.txt | less
