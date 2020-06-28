import sys
import re

sys.path.append('./3')
from q20 import extract_british

text = extract_british()
list = text.split('\n')

for line in list:
    re_ = re.search('\|(.*?)\s=(.*?)$', line)
    if re_ is not None:
        dict = {re_.group(1): re.sub('\[.*\]', '.*', re.sub('\'*', '', re_.group(2)))}
        print(dict)
