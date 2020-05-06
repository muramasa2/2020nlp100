s = "I am an NLPer"

def word_gram(s, N):
    s = s.replace('.', '').replace(',', '')
    words = s.split()
    out=[]

    for i in range(len(words)-N+1):
        out.append(''.join(words[i:i+N]))

    return out

def sentence_gram(s, N):
    s = s.replace(' ', '').replace('.', '').replace(',', '')

    out=[]
    for i in range(len(s)-N):
        out.append(s[i:i+N])

    return out

w_gram = word_gram(s, 2)
s_gram = sentence_gram(s, 2)

print('word gram:', w_gram)
print('sentence gram:', s_gram)
