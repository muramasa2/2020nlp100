from q30 import read_wakati
from collections import Counter

wakati_list = read_wakati()

words = Counter([wakati['surface'] for sentence in wakati_list for wakati in sentence]).most_common()

print(words[:10])
