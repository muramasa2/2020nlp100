import sys
import re

sys.path.append('./3')
from q20 import extract_british

text = extract_british()
list = text.split('\n')
for line in list:
    re_ = re.search('ファイル:(.*?)\|', line)
    if re_ is not None:
        print(re_.group(1))
