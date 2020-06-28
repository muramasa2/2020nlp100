# coding:utf-8
import os

txt_path = '../data/ai.ja.txt'
save_path = '../5/ai.ja.txt.parsed'


class Morph:
    '''
    形態素を表すクラス
    '''
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def is_end_of_sentence(self):
        if self.pos1 == '句点':
            is_eos=True
            return is_eos

    def __str__(self):
        str_ = 'surface: {}, base: {}, pos: {}, pos1: {}'.format(self.surface, self.base, self.pos, self.pos1)
        return str_

def make_morph_list(analyzed_file_name):
    '''
    係り受け解析済み文章ファイルを入力して各文のMorphオブジェクトのリストで返す
    :param analyzed_file_name 係り受け解析済み文書ファイル名
    :return list_ 一つの文をMorphオブジェクトのリストで返したリスト
    '''
    sentences = []
    sentence = []
    with open(analyzed_file_name, 'r', encoding='utf8') as f:
        for line in f:
            line_list = line.split()

            if (line_list[0] == '*') | (line_list[0] == 'EOS'):
                pass
            else:
                line_list = line_list[0].split(',') + line_list[1].split(',')
                _morph = Morph(surface=line_list[0], base=line_list[7], pos=line_list[1], pos1=line_list[2])
                sentence.append(_morph)

                if _morph.is_end_of_sentence():
                    sentences.append(sentence)
                    sentence = []

        return sentences

if __name__ =='__main__':
    import CaboCha

    c = CaboCha.Parser()
    with open(txt_path, 'r', encoding='utf8') as in_f:
        with open(save_path, 'w', encoding='utf8') as out_f:
            for line in in_f:
                c_parse = c.parse(line.lstrip())
                c_dependancy = c_parse.toString(CaboCha.FORMAT_LATTICE)
                out_f.write(c_dependancy)


    morphed_sentences = make_morph_list(save_path)

    for morph in morphed_sentences[2]:
        print(morph)
