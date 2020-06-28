from q41 import make_chunk_list

txt_path = '../5/ai.ja.txt.parsed'
chunked_sentences = make_chunk_list(txt_path)
sentence_idx = 2
# src_word_idx = int(input('input src word idx:'))

def extract_src_target(sentences, sentence_idx, src_word_idx):
    src_word = extract_surface(sentences, sentence_idx, src_word_idx)

    dst_word_idx = sentences[sentence_idx][src_word_idx].dst
    dst_word = extract_surface(sentences, sentence_idx, dst_word_idx)

    output = '\t'.join([src_word, dst_word])
    return output

def extract_surface(sentences, sentence_idx, idx):
    surfaces = [morph.surface for morph in sentences[sentence_idx][idx].morphs]
    word = ''.join(surfaces)

    return word


if __name__  == '__main__':
    for src_word_idx in range(len(chunked_sentences[sentence_idx])-1):
        output = extract_src_target(chunked_sentences, sentence_idx, src_word_idx)
        print(output)
