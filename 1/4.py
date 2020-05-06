import numpy as np

s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
s.replace('.','')
s_list = s.split()

idx = list(np.arange(len(s_list))+1)

first_idx = [1, 5, 6, 7, 8, 9, 15, 16, 19]
second_idx = list(set(idx) - set(first_idx))

first_idx_np = np.array(first_idx) - 1
second_idx_np = np.array(second_idx) - 1

first_list = [s_list[i][:1] for i in first_idx_np]
second_list = [s_list[i][:2] for i in second_idx_np]

out = {}
for idx, word in zip(first_idx, first_list):
    out[idx] = word

for idx, word in zip(second_idx, second_list):
    out[idx] = word

out = sorted(out.items())
print(out)

############
# 模範解答 #
############

def element_symbol(s, number):
    out_dict = {}
    splited = re.split('\s', s)
    for i, w in enumerate(splited, start=1):
        if i in number:
            out_dict[w[:1]] = i
        else:
            out_dict[w[:2]] = i

    return out_dict

sentence = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
only_first_number = [1, 5, 6, 7, 8, 9, 15, 16, 19]
print(element_symbol(sentence, only_first_number))
