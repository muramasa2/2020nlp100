s1 = 'paraparaparadise'
s2 = 'paragraph'

def sentence_gram(s, N):
    s = s.replace(' ', '').replace('.', '').replace(',', '')

    out=[]
    for i in range(len(s)-N):
        out.append(s[i:i+N])

    return out

s1_wgram = sentence_gram(s1, 2)
s2_wgram = sentence_gram(s2, 2)

sum_list = list(set(s1_wgram) | set(s2_wgram))
multi_list = set(s1_wgram) & set(s2_wgram)
sub_list = set(s1_wgram) - set(s2_wgram)

print('和集合:', sum_list)
print('積集合:', multi_list)
print('差集合:', sub_list)
