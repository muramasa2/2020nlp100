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
