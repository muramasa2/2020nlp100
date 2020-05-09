import sys

print(sorted(int(line.split('\t')[2]) for line in sys.stdin)[::-1])

############
# 模範解答 #
############
import sys

sorted_list = sorted(sys.stdin, key=lambda x: int(x.split('\t')[2]), reverse=True)

# sort -k3 -nr popular-names.txt
