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

def is_sahen(chunk_morphs):
    return len(chunk_morphs) == 2 and chunk_morphs[0].pos1 == 'サ変接続' and chunk_morphs[1].surface == 'を'

def split_sahen(src_chunk):
    for i in range(len(src_chunk.morphs)):
        if is_sahen(src_chunk.morphs):
            return str(src_chunk.morphs[i]), src_chunk.morphs[:i] + src_chunk.morphs[i+1:]
    return None, src_chunk.morphs

txt_path = '../5/ai.ja.txt.parsed'
sentences = make_chunk_list(txt_path)

with open('case_pattern.txt', 'w') as f:
    for sentence in sentences:
        for chunk in sentence:
            if get_first_verb(chunk):
                verb = get_first_verb(chunk)
                src_chunk = sentence[chunk.srcs]
                sahen, rest = split_sahen(src_chunk)
                print(src_chunk)
                if sahen and extract_cases(rest):
                    print('True')
                    args = extract_args(src_chunk)
                    line = '{}\t{}\t{}'.format(sahen + verb, ' '.join(cases), ' '.join(args))
                    print(line, file=f)
