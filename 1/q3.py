s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
s = s.replace(',', '').strip('.')
s_list = s.split(' ')

print(list(map(len, s_list)))

############
# 模範解答 #
############

import re
def circumference(s):
    splited = re.split('\s', re.sub(r'[,.], '', s))
    word_len = list(map(len, splited))
    return word_len

sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
print(circumference(sentence))
