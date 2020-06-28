s1 = 'パトカー'
s2 = 'タクシー'

s_list = []
for idx in range(len(s1)):
    s_list.append(s1[idx])
    s_list.append(s2[idx])

out = ''.join(s_list)
print(out)

############
# 模範解答 #
############

def connect_strings(sone, stwo):
    result = "".join(s1+s2 for s1,s2 in zip(sone, stwo))
    return result

print(connect_strings("パトカー", "タクシー"))
