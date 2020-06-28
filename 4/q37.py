from q30 import read_wakati
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

wakati_list = read_wakati()
sentences = [sentence for sentence in wakati_list for wakati in sentence if '猫' in wakati['surface']]

words = Counter([wakati['surface'] for sentence in sentences for wakati in sentence if '猫' != wakati['surface']]).most_common()

word = dict(words[:10]).keys()
freq = dict(words[:10]).values()

fp = FontProperties(fname=r'C:\Windows\Fonts/HGRGE.TTC', size=14)
plt.bar(range(len(freq)), freq)
plt.xticks(range(10), word, fontproperties=fp)
plt.savefig('fig37.jpg')


###########
# 模範解答 #
###########
from collections import defaultdict
import matplotlib.pyplot as plt


def parseMecab(block):
    res = []
    for line in block.split('\n'):
        if line == '':
            return res
        (surface, attr) = line.split('\t')
        attr = attr.split(',')
        lineDict = {
            'surface': surface,
            'base': attr[6],
            'pos': attr[0],
            'pos1': attr[1]
        }
        res.append(lineDict)


def extract(block):
    return [b['base'] for b in block]


filename = 'neko.txt.mecab'
with open(filename, mode='rt', encoding='utf-8') as f:
    blockList = f.read().split('EOS\n')
blockList = list(filter(lambda x: x != '', blockList))
blockList = [parseMecab(block) for block in blockList]
wordList = [extract(block) for block in blockList]
wordList = list(filter(lambda x: '猫' in x, wordList))
d = defaultdict(int)
for word in wordList:
    for w in word:
        if w != '猫':
            d[w] += 1
ans = sorted(d.items(), key=lambda x: x[1], reverse=True)[:10]
labels = [a[0] for a in ans]
values = [a[1] for a in ans]

fp = FontProperties(fname=r'C:\Windows\Fonts/HGRGE.TTC', size=14)
plt.figure(figsize=(8, 8))
plt.barh(labels, values)
plt.yticks(range(len(labels)), labels, fontproperties=fp)
plt.show()
