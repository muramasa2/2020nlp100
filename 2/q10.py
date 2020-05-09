# import sys
#
# i=0
# for line in sys.stdin:
#     i+=1
# print(i)


import sys

i=0
for i, _ in enumerate(sys.stdin, start=1):
    pass

print(i)

# wc -l < popular-names.txt
