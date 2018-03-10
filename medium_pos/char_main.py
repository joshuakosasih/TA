import numpy as np
from DataProcessor import DataIndexer as DI
from DataProcessor import DataLoader as DL
from DataProcessor import DataMapper as DM
from keras import Model
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Lambda
from keras.utils import plot_model
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf

from keras_contrib.layers import CRF

"""
Preparing file
"""

train = DL('id-ud-train')
test = DL('id-ud-test')

"""
Create Word & Label Index
"""

char = DI(train.words + test.words)
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', char.cnt - 1, 'unique chars.'
print 'Found', label.cnt - 1, 'unique labels.'

"""
Load pre-trained embedding
"""

char_embeddings_index = {}
CE_DIR = raw_input('Enter embedding file name: ')

print 'Loading', CE_DIR, '...'
f = open(CE_DIR, 'r')
for line in f:
    values = line.split()
    chars = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    char_embeddings_index[chars] = coefs

f.close()

print('Found %s char vectors.' % len(char_embeddings_index))

CHAR_EMBEDDING_DIM = len(coefs)

char_notfound = []  # list kata yang tidak terdapat dalam embedding
char_embedding_matrix = np.zeros((len(char.index) + 1, int(CHAR_EMBEDDING_DIM)))
for chars, i in char.index.items():
    embedding_vector = char_embeddings_index.get(chars)
    if embedding_vector is not None:
        char_embedding_matrix[i] = embedding_vector
    else:
        char_notfound.append(chars)

print('%s unique chars not found in embedding.' % len(char_notfound))

"""
Converting text data to int using index
"""
padsize = 188

x_test_tmp1 = []
char_padsize = 0
for sent in train.words:
    x_map = DM(sent, char.index)
    if x_map.padsize > char_padsize:
        char_padsize = x_map.padsize
    x_test_tmp1.append(x_map)

# x_test = DM(test.words[0], char.index)
# char_padsize = max([x_train.char_padsize, x_test.char_padsize])

x_test_tmp2 = []
for sent in x_test_tmp1:
    sent.pad(char_padsize)
    x_test_tmp2.append(sent.padded)

print('Padded until %s chars.' % char_padsize)

zeroes = []
for i in range(char_padsize):
    zeroes.append(0)

x_train = []
for sent in x_test_tmp2:
    padded_sent = sent
    pad = padsize - len(sent)
    for i in range(pad):
        padded_sent = np.vstack((zeroes, padded_sent))
    x_train.append(padded_sent)

print('Padded until %s tokens.' % padsize)

y_train = DM(train.labels, label.index)
y_test = DM(test.labels, label.index)

y_train.pad(padsize)

y_encoded = to_categorical(y_train.padded, num_classes=len(label.index) + 1)

"""
Create keras model
"""


def reshape_one(c):
    return K.reshape(c, (tf.shape(c)[0] * padsize, char_padsize, CHAR_EMBEDDING_DIM))


def reshape_two(c):
    return K.reshape(c, (tf.shape(c)[0] / padsize, padsize, len(label.index) + 1))


MAX_WORD_LENGTH = char_padsize

embedding_layer_c = Embedding(len(char.index) + 1,
                              CHAR_EMBEDDING_DIM,
                              weights=[char_embedding_matrix],
                              input_length=MAX_WORD_LENGTH,
                              trainable=False)

sequence_input_c = Input(shape=(padsize, MAX_WORD_LENGTH,), dtype='int32')

embedded_sequences_c = embedding_layer_c(sequence_input_c)

rone = Lambda(reshape_one)(embedded_sequences_c)

merge_m = raw_input('Enter merge mode: ')
gru_karakter = Bidirectional(GRU(CHAR_EMBEDDING_DIM, return_sequences=False), merge_mode=merge_m, weights=None)(rone)

preds = Dense(len(label.index) + 1, activation='softmax')(gru_karakter)

rtwo = Lambda(reshape_two)(preds)

model = Model(sequence_input_c, rtwo)

model.summary()
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])

plot_model(model, to_file='model.png')

epoch = 2
batch = 8
model.fit(np.array(x_train), np.array(y_encoded), epochs=epoch, batch_size=batch)

"""
Converting text data to int using index
"""

x_test_tmp1 = []
for sent in test.words:
    x_map = DM(sent, char.index)
    if x_map.padsize > char_padsize:
        char_padsize = x_map.padsize
    x_test_tmp1.append(x_map)

x_test_tmp2 = []
for sent in x_test_tmp1:
    sent.pad(char_padsize)
    x_test_tmp2.append(sent.padded)

print('Padded until %s chars.' % char_padsize)

zeroes = []
for i in range(char_padsize):
    zeroes.append(0)

x_test = []
for sent in x_test_tmp2:
    padded_sent = sent
    pad = padsize - len(sent)
    for i in range(pad):
        padded_sent = np.vstack((zeroes, padded_sent))
    x_test.append(padded_sent)

print('Padded until %s tokens.' % padsize)

"""
Evaluate
"""

mateval = []
for labr in range(label.cnt):
    row = []
    for labc in range(label.cnt):
        row.append(0)
    mateval.append(row)

print "Computing..."
raw_results = model.predict(np.array(x_test))
results = []
for raw_result in raw_results:
    result = []
    for token in raw_result:
        value = np.argmax(token)
        result.append(value)
    results.append(result)

y_test.pad(padsize)
total_nonzero = 0  # to get labelled token total number
for i, sent in enumerate(y_test.padded):
    for j, token in enumerate(sent):
        pred = results[i][j]
        answ = token
        mateval[answ][pred] = mateval[answ][pred] + 1  # row shows label and column shows prediction given
        if not answ == 0:
            total_nonzero = total_nonzero + 1

total_true = 0
for i in range(1, len(mateval)):
    total_true = total_true + mateval[i][i]

total_false = total_nonzero - total_true

print "True", total_true
print "False", total_false
print "True percentage", float(total_true) / float(total_nonzero)

"""
Predict function
"""

# pm.predict('buah hati dia ingin memiliki cinta seorang anak tetapi aku tidak cinta kemudian menikah untuk kedua', char_padsize)
