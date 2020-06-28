from q30 import read_wakati
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
wakati_list = read_wakati()

words = Counter([wakati['surface'] for sentence in wakati_list for wakati in sentence]).most_common()

words_cnt = [int(word[1]) for word in words[:10]]
words_ctx = [word[0] for word in words[:10]]

fp = FontProperties(fname=r'C:\Windows\Fonts/HGRGE.TTC', size=14)
plt.bar(range(10), words_cnt)
plt.xticks(range(10), words_ctx, fontproperties=fp)
plt.savefig('fig36.jpg')


###########
# 模範解答 #
###########
from collections import Counter
import matplotlib.pyplot as plt
from q0 import get_neko_morphemes

morphemes_list = get_neko_morphemes()

words = Counter([morpheme["base"] for morphemes in morphemes_list for morpheme in morphemes]).most_common()
word_name, word_count = list(zip(*words[:10]))

plt.rcParams["font.family"] = "IPAexGothic"
plt.bar(range(10), word_count, tick_label=word_name)
plt.savefig("fig36.png")
