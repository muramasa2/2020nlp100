import sys

sys.path.append('./3')
from q20 import extract_british

text = extract_british()
list = text.split('\n')
for line in list:
    if 'Category' in line:
        print(line)
