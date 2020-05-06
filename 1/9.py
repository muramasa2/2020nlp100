import random
s = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

words = s.split()
for idx, word in enumerate(words):
    if len(word)>4:
        print(word[1:-1])
        print(''.join(random.sample(word[1:-1], len(word[1:-1]))))
        shuffle = ''.join(random.sample(word[1:-1], len(word[1:-1])))
        words[idx] = shuffle

new_s = ' '.join(words)
print(new_s)
