import sys
import re

sys.path.append('./3')
from q20 import extract_british

text = extract_british()
list = text.split('\n')

for line in list:
    re_ = re.search('\|(.*?)\s=(.*?)$', line)
    if re_ is not None:
        match_temp1 = re.sub('\'*', '', re_.group(2)) # 強調削除
        match_temp2 = re.sub('\[\[(.*?)([|\|].*)?\]\]','\\1',match_temp1)
        match_temp3 = re.sub('<.*>', '', match_temp2)
        dict = {re_.group(1): match_temp3}
        print(dict)
