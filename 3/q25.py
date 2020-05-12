import sys
import re

sys.path.append('./3')
from q20 import extract_british


def baseinf():
    text = extract_british()
    list = text.split('\n')
    dict = {}
    for line in list:
        re_ = re.search('\|(.*?)\s=(.*?)$', line)
        if re_ is not None:
            dict[re_.group(1)]  = re_.group(2)
            # print(dict)
    return dict
