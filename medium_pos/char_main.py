import numpy as np
import PredictModule as pm
from DataProcessor import DataIndexer as DI
from DataProcessor import DataLoader as DL
from DataProcessor import DataMapper as DM
from keras import Model
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Reshape
from keras.utils import plot_model
from keras.utils import to_categorical
from keras_contrib.layers import CRF

"""
Preparing file
"""

train = DL('id-ud-train')
test = DL('id-ud-test')

"""
Create Word & Label Index
"""

char = DI(train.words)
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', char.cnt - 1, 'unique chars.'
print 'Found', label.cnt - 1, 'unique labels.'

"""
Load pre-trained embedding
"""

char_embeddings_index = {}
CE_DIR = 'polyglot-char.txt'

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

x_train_tmp1 = []
char_padsize = 0
for sent in train.words:
    x_map = DM(sent, char.index)
    if x_map.padsize > char_padsize :
        char_padsize = x_map.padsize
    x_train_tmp1.append(x_map)

# x_test = DM(test.words[0], char.index)

# char_padsize = max([x_train.char_padsize, x_test.char_padsize])

x_train_tmp2 = []
for sent in x_train_tmp1:
    sent.pad(char_padsize)
    x_train_tmp2.append(sent.padded)

print('Padded until %s chars.' % char_padsize)

zeroes = []
for i in range(char_padsize):
    zeroes.append(0)

x_train = []
for sent in x_train_tmp2:
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

MAX_SEQUENCE_LENGTH = char_padsize

embedding_layer = Embedding(len(char.index) + 1,
                            CHAR_EMBEDDING_DIM,
                            weights=[char_embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(padsize, MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

reshape_layer = Reshape(padsize, CHAR_EMBEDDING_DIM*char_padsize)(embedded_sequences)

gru_karakter = Bidirectional(GRU(CHAR_EMBEDDING_DIM, return_sequences=False), merge_mode='concat', weights=None)(
    embedded_sequences)

preds = Dense(len(label.index) + 1, activation='softmax')(gru_karakter)

model = Model(sequence_input, preds)

model.summary()
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])

plot_model(model, to_file='model.png')

epoch = 10
batch = 1
model.fit(np.array(x_train), np.array(y_encoded), epochs=epoch, batch_size=batch)

"""
Evaluate
"""

mateval = []
for labr in range(label.cnt):
    row = []
    for labc in range(label.cnt):
        row.append(0)
    mateval.append(row)

x_test.pad(char_padsize)
results = []
print "Computing..."
result = []
for token in raw_results:
    value = np.argmax(token)
    result.append(value)

y_test.pad(char_padsize)
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
print "True percentage", float(total_true)/float(total_nonzero)
"""
Predict function
"""

# pm.predict('buah hati dia ingin memiliki cinta seorang anak tetapi aku tidak cinta kemudian menikah untuk kedua', char_padsize)
