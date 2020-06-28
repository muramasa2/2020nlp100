# -*- coding: utf-8 -*-
# coding:utf-8

from pydotplus import Dot, Edge, Node
from PIL import Image
from q41 import make_chunk_list
from q42 import extract_surface

txt_path = '../5/ai.ja.txt.parsed'
sentences = make_chunk_list(txt_path)

sentence_idx = 7
sentence = sentences[sentence_idx]

graph = Dot(graph_type = 'digraph')
graph.set_fontname('MS Gothic')

# make Node and Edge
for id, chunk in enumerate(sentence):
    word = extract_surface(sentences, sentence_idx, id)
    node = Node(id, label = word)
    graph.add_node(node)
    if chunk.dst != -1:
        edge = Edge(id, chunk.dst)
        graph.add_edge(edge)


graph.write_png('sentence.png')
Image.open('sentence.png')
