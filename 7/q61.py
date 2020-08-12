from gensim.models import KeyedVectors

dataset_path = './data/GoogleNews-vectors-negative300.bin'

model = KeyedVectors.load_word2vec_format(dataset_path, binary=True)

model.similarity('United_States', 'U.S.')
