import sys
from collections import Counter

col1_freq = Counter(line.split('\t')[0] for line in sys.stdin)

for elem, num in col1_freq.most_common():
    print(num, elem)

# cut -f1 popular-names.txt | sort | uniq -c | sort -nr
