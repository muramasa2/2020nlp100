def sentence_gram(s, N):
    s = s.replace('.', '').replace(',', '')

    out=[]
    for i in range(len(s)-N+1):
        out.append(s[i:i+N])

    return out

def cipher(s):
    s_list = sentence_gram(s, 1)

    out = []
    for w in s_list:
        if w.islower:
            out.append(chr(219 - ord(w)))
        else:
            out.append(w)
    out = ''.join(out)
    return out

str = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

out = cipher(str)
print(out)
d_out = cipher(out)
print(d_out)

############
# 模範解答 #
############
# def chipher(s):
#     result = ""
#     for character in s:
#         result += re.sub(r'[a-z]', chr(219 - ord(character)), character)
#     return result

def chipher(src):
    return re.sub(r'[a-z]', lambda m: chr(219 - ord(m.group(0))), src)

sentence = "Hi, Thank you for reading my article!!"
print(chipher(sentence))
print(chipher(chipher(sentence)))
