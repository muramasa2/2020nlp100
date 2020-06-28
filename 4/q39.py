from q30 import read_wakati
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
wakati_list = read_wakati()

words = Counter([wakati['surface'] for sentence in wakati_list for wakati in sentence]).most_common()

words_cnt = [int(word[1]) for word in words]
words_ctx = [word[0] for word in words]

plt.plot(range(1, len(words_cnt)+1), words_cnt)
plt.xscale('log')
plt.yscale('log')
plt.savefig('fig39.jpg')


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
plt.plot(list(range(1, len(word_count) + 1)), word_count)
plt.xscale("log")
plt.yscale("log")
plt.savefig("fig39.png")
