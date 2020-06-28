from q41 import make_chunk_list


def get_first_verb(chunk):
    morphs = chunk.morphs
    for morph in morphs:
        if morph.pos =='動詞':
            return morph.base


def get_last_case(chunk):
    morphs = chunk.morphs
    for morph in morphs[::-1]:
        if morph.pos =='助詞':
            return morph.surface


def extract_cases(src_chunk):
    xs = get_last_case(src_chunk)
    return xs

def extract_args(src_chunk):
    if get_last_case(src_chunk):
        xs = get_last_case(src_chunk)
    return xs

txt_path = '../5/ai.ja.txt.parsed'
sentences = make_chunk_list(txt_path)

sentence_idx = 7
sentence = sentences[sentence_idx]

with open('case_pattern.txt', 'w') as f:
    for sentence in sentences:
        for chunk in sentence:
            if get_first_verb(chunk):
                verb = get_first_verb(chunk)
                src_chunk = sentence[chunk.srcs]
                if extract_cases(src_chunk):
                    cases = extract_cases(src_chunk)
                    args = extract_args(src_chunk)
                    line = '{}\t{}\t{}'.format(verb, ' '.join(cases), ' '.join(args))
                    print(line, file=f)
