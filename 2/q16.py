# from q14 import arg_lines
# import sys
#
#
# def split_line(N):
#     for line in sys.stdin:
#         lines = line.rstrip('.,\n').split('\t')
#         length = len(lines)
#         n_lines = [' '.join(lines[(length*i//N):(length*(i+1)//N)]) for i in range(N)]
#         print(n_lines)
#
# if __name__ == '__main__':
#     split_line(arg_lines())

############
# 模範解答 #
############
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description='Output pices of FILE to FILE1, FILE2, ...;')
    parser.add_argument('file')
    parser.add_argument('-n', '--number', type=int,
                        help='split FILE into n pieces')
    args = parser.parse_args()
    file_split(args.file, args.number)


def file_split(filename, N):
    with open(filename) as f:
        n_lines = sum(1 for _ in f)
        f.seek(0)
        for nth, width in enumerate((n_lines+i)//N for i in range(N)):
            with open(f'{filename}.split{nth}', 'w') as fo:
                for _ in range(width):
                    fo.write(f.readline())

if __name__ == '__main__':
    main()

# split -n 1/5 -d popular-names.txt popular-names.txt
