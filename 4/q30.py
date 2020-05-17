import os
import MeCab as me


def create_wakati():
    m = me.Tagger('')

    with open('neko.txt', 'r', encoding='utf8') as f:
        sentence = f.read()

    print(m.parse(sentence))

    out = 'neko.txt.mecab'

    with open(out, 'w', encoding='utf-8') as f:
        f.write(m.parse(sentence))


def read_wakati(path='neko.txt.mecab'):
    with open(path, 'r', encoding='utf-8') as f:
        wakati_dict = f.read().split('\n')
        # 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音

    sentence = []
    sentences = []
    for line in wakati_dict:
        word = line.split('\t')

        if len(word) > 1:
            surface = word[0]
            result = word[1].split(',')

            wakati_dict = {
                'surface': surface,
                'base': result[6],
                'pos': result[0],
                'pos1': result[1]
                }

            sentence.append(wakati_dict)

            if wakati_dict['pos1']=='句点' or wakati_dict['pos1']=='空白':
                sentences.append(sentence)
                sentence=[]

    return sentences

if __name__ =='__main__':
    wakati_path = 'neko.txt.mecab'
    if os.path.exists(wakati_path) is None:
        create_wakati()
        print('create wakati!')

    wakati_sentence = read_wakati(wakati_path)
    print(wakati_sentence)
