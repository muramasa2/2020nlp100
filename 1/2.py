s1 = 'パトカー'
s2 = 'タクシー'

s_list = []
for idx in range(len(s1)):
    s_list.append(s1[idx])
    s_list.append(s2[idx])

out = ''.join(s_list)
print(out)
