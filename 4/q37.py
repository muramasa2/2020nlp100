from q30 import read_wakati
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

wakati_list = read_wakati()
sentences = [sentence for sentence in wakati_list for wakati in sentence if '猫' in wakati['surface']]

words = Counter([wakati['surface'] for sentence in sentences for wakati in sentence if '猫' != wakati['surface']]).most_common()

word = dict(words).keys()
freq = dict(words).values()
fp = FontProperties(fname=r'C:\Windows\Fonts/HGRGE.TTC', size=14)
plt.bar(range(len(freq), freq))
plt.xticks(word, fontproperties=fp)
