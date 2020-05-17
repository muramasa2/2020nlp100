from q30 import read_wakati
from collections import Counter

wakati_list = read_wakati()

words = Counter([wakati['surface'] for sentence in wakati_list for wakati in sentence]).most_common()

print(words[:10])

###########
# 模範解答 #
###########
from collections import Counter
from q30 import get_neko_morphemes

morphemes_list = get_neko_morphemes()

words = Counter([morpheme["base"] for morphemes in morphemes_list for morpheme in morphemes]).most_common()
print(words[:10])
