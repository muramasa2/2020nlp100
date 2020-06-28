import sys
import re

sys.path.append('./3')
from q20 import extract_british

text = extract_british()
list = text.split('\n')

for line in list:
    re_line = re.search('Category:(.*?)(|\|.*)]', line)

    if re_line is not None:
        print(re_line.group(1))
