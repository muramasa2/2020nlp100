from q41 import make_chunk_list
from q48 import isIncludingnoun, extract_surface

txt_path = '../5/ai.ja.txt.parsed'
sentences = make_chunk_list(txt_path)

sentence = sentences[25]

def extract_path(x, y, sent):
    xs = []
    ys = []
    while x != y:
        if x < y:
            xs.append(x)
            x = sent[x].dst
        else:
            ys.append(y)
            y = sent[y].dst

    return xs, ys, x

def remove_initial_nouns(chunk):
    for i, morph in enumerate(chunk.morphs):
        if morph.pos != '名詞':
            break

    return ''.join([str(morph.surface) for morph in chunk.morphs[i:]]).strip()

def path_to_str(xs, ys, last, sent):
    xs_surface = [extract_surface(sent[x]) for x in xs]
    ys_surface = [extract_surface(sent[y]) for x in ys]
    last_surface = extract_surface(sent[last])
    if xs and ys:
        xs = ['X' + remove_initial_nouns(sentence[xs[0]])] + [x for x in xs_surface[1:]]
        ys = ['Y' + remove_initial_nouns(sentence[ys[0]])] + [y for y in ys_surface[1:]]
        last = str(last_surface)
        return ' -> '.join(xs) + ' | ' + ' -> '.join(ys) + ' | ' + last

    else:
        xs = xs + ys
        xs = ['X' + remove_initial_nouns(sentence[xs[0]])] + [x for x in xs_surface[1:]]
        last = 'Y' + remove_initial_nouns(sentence[last])
        return ' -> '.join(xs + [last])


heads = [n for n in range(len(sentence)) if isIncludingnoun(sentence[n])]
print('パスの先頭:', heads)

pairs = [
    (heads[n], second)
    for n in range(len(heads))
    for second in heads[n+1:]
]
print('パスの先頭のペア:', pairs)

print('係り受けのパス:')
for x, y in pairs:
    x_path, y_path, last = extract_path(x, y, sentence)
    if x < len(sentence) and y < len(sentence) and last < len(sentence):
        path = path_to_str(x_path, y_path, last, sentence)
    print(path)
