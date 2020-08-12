from gensim.models import KeyedVectors

dataset_path = './data/GoogleNews-vectors-negative300.bin'

model = KeyedVectors.load_word2vec_format(dataset_path, binary=True)

similarity_list  = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'])

similarity_list
