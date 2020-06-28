from q41 import make_chunk_list


txt_path = '../5/ai.ja.txt.parsed'
sentences = make_chunk_list(txt_path)

sentence = sentences[25]

def isIncludingnoun(chunk):
    for morph in chunk.morphs:
        return morph.pos == '名詞'

def extract_surface(chunk):
    surfaces = [morph.surface for morph in chunk.morphs]
    words = ''.join(surfaces)

    return words

def isExistedDst(sentence, dst):
    return sentence[dst] != None

def extract_dstchunk(sentence, src_chunk):
    path = []
    dst = src_chunk.dst
    while dst != -1 and dst < len(sentence):
        path.append(dst)
        dst = sentence[dst].dst

    return path

if __name__ == '__main__':
    for chunk in sentence:
        dst_sentences = []
        if isIncludingnoun(chunk):
            dst_chunkId = chunk.dst
            src_sentence = extract_surface(chunk)
            output=src_sentence
            for dstId in extract_dstchunk(sentence, chunk):
                dst_sentence = extract_surface(sentence[dstId])
                output = ' -> '.join([output, dst_sentence])
            print(output)
