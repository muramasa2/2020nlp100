import random
s = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

words = s.split()
for idx, word in enumerate(words):
    if len(word)>4:
        print(word[1:-1])
        print(''.join(random.sample(word[1:-1], len(word[1:-1]))))
        shuffle = ''.join(random.sample(word[1:-1], len(word[1:-1])))
        words[idx] = word[0] + shuffle + word[-1]

new_s = ' '.join(words)
print(new_s)

############
# 模範解答 #
############
import random

def mixing_word(sentence):
    splited = sentence.split(" ")
    randomed_list = [s[0] + ''.join(random.sample(s[1:-1], len(s)-2))+ s[-1] if len(s) >= 4 else s for s in splited]
    return " ".join(randomed_list)

input_text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
mixing_word(input_text)
