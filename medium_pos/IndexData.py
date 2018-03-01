import os
import nltk
import numpy as np
from nltk.tag import StanfordNERTagger

"""
Preparing file
"""

name = 'id-ud-train'

print "Openning file", name
myfile = open(name + '.pos', 'r')

print "Processing..."
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

words = []
labels = []

for sent in corpus:
    line = []
    y_true = []
    for token in sent:
        line.append(token[0])
        y_true.append(token[1])
    words.append(line)
    labels.append(y_true)

"""
Create Word & Label Index
"""

w = 1
word_index = {}
for sent in words:
    for token in sent:
        if token not in word_index:
            word_index[token] = w
            w = w + 1

l = 1
labels_index = {}
for sent in labels:
    for token in sent:
        if token not in labels_index:
            labels_index[token] = l
            l = l + 1

print('Found %s unique words.' % len(word_index))
print('Found %s unique labels.' % len(labels_index))

"""
Load pre-trained embedding
"""

embeddings_index = {}
GLOVE_DIR = 'WE_w.txt'

f = open(GLOVE_DIR, 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 200

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

"""
Converting text data to int using index
"""

x = []
for sent in words:
    x_token = []
    for token in sent:
        x_token.append(word_index[token])
    x.append(x_token)

x_padded = pad_sequences(x)
print('Padded until %s tokens.' % len(x_padded[0]))

y = []
for sent in labels:
    y_token = []
    print sent
    for token in sent:
        y_token.append(labels_index[token])
    y.append(y_token)

y_padded = pad_sequences(y) # label for evaluation
y_encoded = to_categorical(y_padded)

"""
y_trimmed = []
for sent in y_encoded:
    y_token = []
    for token in sent:
        y_token.append(token)
    y_trimmed.append(y_token)

y_trimmed = np.array(y_trimmed)
"""