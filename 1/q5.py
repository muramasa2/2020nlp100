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

############
# 模範解答 #
############

def generate_ngram(sentence, N):
    return [sentence[i:i+N] for i in range(len(sentence)-N+1)]

input_text = "I am an NLPer"

print("単語bi-garm : ", str(generate_ngram(input_text.split(' '), 2)))
print("文字bi-garm : ", str(generate_ngram(input_text.split(' '), 2)))
