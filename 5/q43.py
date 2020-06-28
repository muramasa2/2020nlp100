from q41 import make_chunk_list
from q42 import extract_src_target

txt_path = '../5/ai.ja.txt.parsed'
sentences = make_chunk_list(txt_path)
sentence_idx = 2


def check_poses(sentences, sentence_idx, word_idx, target_pos):
    poses = [morph.pos for morph in sentences[sentence_idx][word_idx].morphs]
    if target_pos in poses:
        return True
    else:
        return False


for src_word_idx in range(len(sentences[sentence_idx])-1):
    if check_poses(sentences, sentence_idx, src_word_idx, '名詞'):
        dst_word_idx = sentences[sentence_idx][src_word_idx].dst

        if check_poses(sentences, sentence_idx, dst_word_idx, '動詞'):
            print(extract_src_target(sentences, sentence_idx, src_word_idx))
