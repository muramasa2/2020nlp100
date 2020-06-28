from q30 import read_wakati

wakati_list = read_wakati()

verb_dict=[]
for sentence in wakati_list:
    for wakati_dict in sentence:
        if wakati_dict['pos']=='動詞':
            value = list(wakati_dict.values())
            verb_dict.append(value[0])

print(verb_dict)


###########
# 模範解答 #
###########
from q30 import get_neko_morphemes

morphemes_list = get_neko_morphemes()

result = []

for morphemes in morphemes_list:
    for morpheme in morphemes:
        if morpheme["pos"] == "動詞":
            result.append(morpheme["surface"])

print(result[:10])
