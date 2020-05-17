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
