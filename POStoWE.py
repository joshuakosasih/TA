import os
import nltk
import numpy as np
from nltk.tag import StanfordNERTagger
from keras.layers import Embedding
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

"""
Preparing file
"""

name = raw_input('Enter file name: ')

myfile = open(name + '.pos', 'r')

mydict = []

for line in myfile:
    mydict.append(line)

corpus = []
lines = []

for line in mydict:
    if line == '-----\r\n':
        corpus.append(lines)
        lines = []
    else:
        lines.append((nltk.word_tokenize(line)[0], nltk.word_tokenize(line)[1]))

lines = []
y_true = []

for sent in corpus:
    line = []
    for token in sent:
        line.append(token[0])
        y_true.append(token[1])
    lines.append(line)

"""
Create Word Index
"""

w = 1
word_index = {}
for sent in lines:
    for token in sent:
        if token not in word_index:
            word_index[token] = w
            w = w + 1

print('Found %s unique words.' % len(word_index))

"""
Load pre-trained embedding
"""

embeddings_index = {}
GLOVE_DIR = raw_input('Enter pre-trained embedding file name: ')

f = open(GLOVE_DIR, 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = raw_input('Enter embedding dimensions: ')

notfound = 0
notfoundwords = [] # list kata yang tidak terdapat dalam embedding
embedding_matrix = np.zeros((len(word_index) + 1, int(EMBEDDING_DIM)))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        notfound = notfound + 1
        notfoundwords.append(word)

print('%s words not found in embedding.' % notfound)

MAX_SEQUENCE_LENGTH = raw_input('Enter max seq length: ')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

