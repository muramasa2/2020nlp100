from q30 import read_wakati
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
wakati_list = read_wakati()

words = Counter([wakati['surface'] for sentence in wakati_list for wakati in sentence]).most_common()

words_cnt = [int(word[1]) for word in words]
words_ctx = [word[0] for word in words]

plt.hist(words_cnt, bins=50, range=(0, 50))
plt.savefig('fig38.jpg')


###########
# 模範解答 #
###########
from collections import Counter
import matplotlib.pyplot as plt
from q30 import get_neko_morphemes

morphemes_list = get_neko_morphemes()

words = Counter([morpheme["base"] for morphemes in morphemes_list for morpheme in morphemes]).most_common()
_, word_count = list(zip(*words))

plt.rcParams["font.family"] = "IPAexGothic"
plt.hist(word_count, bins=50, range=(1, 50))
plt.savefig("fig38.png")
