from q14 import arg_lines
import sys
from collections import deque

def tail(N):
    buf = deque(sys.stdin, N)
    print(''.join(buf))

if __name__ =='__main__':
    tail(arg_lines())

# tail -n 10 popular-names.txt
