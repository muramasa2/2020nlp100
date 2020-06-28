s1 = 'paraparaparadise'
s2 = 'paragraph'

def sentence_gram(s, N):
    s = s.replace(' ', '').replace('.', '').replace(',', '')

    out=[]
    for i in range(len(s)-N+1):
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

print('seがXに含まれるかどうか', 'se' in s1_wgram)
print('seがYに含まれるかどうか', 'se' in s2_wgram)

############
# 模範解答 #
############
X_text ="paraparaparadise"
Y_text = "paragraph"

X = set(generate_ngram(X_text, 2))
Y = set(generate_ngram(Y_text, 2))

print("和集合 : " + str(X.union(Y)))
print("積集合 : " + str(X.intersection(Y)))
print("差集合 : " + str(X.difference(Y)))

print("seがXに含まれるか : " + str('se' in X))
print("seがYに含まれるか : " + str('se' in Y))
