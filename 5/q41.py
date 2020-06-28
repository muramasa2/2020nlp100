# coding:utf-8
from q40 import Morph

txt_path = '../5/ai.ja.txt.parsed'

class Chunk:
    '''
    read the data analyzed by CaboCha and express as Chunk object list
    :param morphs   形態素のリスト
           dst      係り先文節インデックス番号
           srcs     係り元文節インデックス番号のリスト
    :return list_
    '''

    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs
        self.dst = int(dst.strip('D'))
        self.srcs = int(srcs)

    def __str__(self):
        str_ = 'srcs: {}, dst: {}, morphs: ({})'.format(self.srcs, self.dst, ' / '.join([str(_morph) for _morph in self.morphs]))
        return str_

def make_chunk_list(analyzed_file_name):
    '''
    係り受け解析済み文章ファイルを入力して各文のChunkオブジェクトのリストで返す
    '''
    sentences = []
    sentence = []
    _chunk = None
    with open(analyzed_file_name, 'r', encoding='utf8') as f:
        for line in f:
            line_list = line.split()

            if line_list[0] == '*':
                if _chunk is not None:
                    sentence.append(_chunk)
                _chunk = Chunk(morphs=[], dst=line_list[2], srcs=line_list[1])
            elif line_list[0] == 'EOS':
                if _chunk is not None:
                    sentences.append(sentence)
                if len(sentence) > 0:
                    sentences.append(sentence)
                _chunk = None
                sentence = []

            else:
                line_list = line_list[0].split(',') + line_list[1].split(',')
                _morph = Morph(surface=line_list[0], base=line_list[7], pos=line_list[1], pos1=line_list[2])
                _chunk.morphs.append(_morph)

    return sentences

if __name__ == '__main__':
    chunked_sentences = make_chunk_list(txt_path)

    for chunk in chunked_sentences[2]:
        print(chunk)
