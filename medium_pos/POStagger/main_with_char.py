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
from keras_contrib.layers import CRF
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

trainable = True  # embedding trainable or not
mask = True  # mask pad (zeros) or not

"""
Preparing file
"""

train = DL('id-ud-train.pos')
test = DL('id-ud-test.pos')
train.add('id-ud-dev.pos')

"""
Load pre-trained word embedding
"""

embeddings_index = {}
WE_DIR = 'polyglot.vec'

print 'Loading', WE_DIR, '...'
f = open(WE_DIR, 'r')
for line in f:
    values = line.split()
    wrd = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[wrd] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

we_words = []
for wrd in embeddings_index:
    we_words.append(wrd)

"""
Load pre-trained char embedding
"""

char_embeddings_index = {}
CE_DIR = 'polyglot-char.vec'

print 'Loading', CE_DIR, '...'
f = open(CE_DIR, 'r')
for line in f:
    values = line.split()
    chars = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    char_embeddings_index[chars] = coefs

f.close()

print('Found %s char vectors.' % len(char_embeddings_index))

ce_words = []
for chr in char_embeddings_index:
    ce_words.append(chr)

"""
Create Word & Label Index
"""

char = DI(train.words + ce_words)
char.save('char')
word = DI([train.words, [we_words]])
word.save('word')
label = DI([train.labels])

print 'Found', word.cnt - 1, 'unique words.'
print 'Found', char.cnt - 1, 'unique chars.'
print 'Found', label.cnt - 1, 'unique labels.'

"""
Create word embedding matrix
"""

EMBEDDING_DIM = len(coefs)

notfound = []
embedding_matrix = np.zeros((len(word.index) + 1, int(EMBEDDING_DIM)))
for wrd, i in word.index.items():
    embedding_vector = embeddings_index.get(wrd)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        notfound.append(wrd)

print('%s unique words not found in embedding.' % len(notfound))

"""
Create char embedding matrix
"""

CHAR_EMBEDDING_DIM = len(coefs)

char_notfound = []
char_embedding_matrix = np.zeros((len(char.index) + 1, int(CHAR_EMBEDDING_DIM)))
for chars, i in char.index.items():
    embedding_vector = char_embeddings_index.get(chars)
    if embedding_vector is not None:
        char_embedding_matrix[i] = embedding_vector
    else:
        char_notfound.append(chars)

print('%s unique chars not found in embedding.' % len(char_notfound))

"""
Converting word text data to int using index
"""
trimlen = 160
train.trim(trimlen)
test.trim(trimlen)

x_train = DM(train.words, word.index)
x_test = DM(test.words, word.index)
print "Number of OOV:", len(x_test.oov_index)
print "OOV word occurences:", x_test.oov

padsize = max([x_train.padsize, x_test.padsize])
x_train.pad(padsize)
print('Padded until %s tokens.' % padsize)

y_train = DM(train.labels, label.index)
y_test = DM(test.labels, label.index)

y_train.pad(padsize)
y_encoded = to_categorical(y_train.padded)

"""
Converting char text data to int using index
"""

x_train_tmp1 = []
char_padsize = 0
for sent in train.words:
    x_map = DM(sent, char.index, False)
    if x_map.padsize > char_padsize:
        char_padsize = x_map.padsize
    x_train_tmp1.append(x_map)

x_train_tmp2 = []
for sent in x_train_tmp1:
    sent.pad(char_padsize)
    x_train_tmp2.append(sent.padded)

print('Padded until %s chars.' % char_padsize)

zeroes = []
for i in range(char_padsize):
    zeroes.append(0)

x_train_char = []
for sent in x_train_tmp2:
    padded_sent = sent
    pad = padsize - len(sent)
    for i in range(pad):
        padded_sent = np.vstack((zeroes, padded_sent))
    x_train_char.append(padded_sent)

print('Padded until %s tokens.' % padsize)

"""
Create keras word model
"""
MAX_SEQUENCE_LENGTH = padsize

embedding_layer = Embedding(len(word.index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=trainable,
                            mask_zero=mask)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

"""
Create keras char model
"""


def reshape_one(c):
    return K.reshape(c, (tf.shape(c)[0] * padsize, char_padsize, CHAR_EMBEDDING_DIM))


def reshape_two(c):
    return K.reshape(c, (tf.shape(c)[0] / padsize, padsize, CHAR_EMBEDDING_DIM))


MAX_WORD_LENGTH = char_padsize

embedding_layer_c = Embedding(len(char.index) + 1,
                              CHAR_EMBEDDING_DIM,
                              weights=[char_embedding_matrix],
                              input_length=MAX_WORD_LENGTH,
                              trainable=trainable,
                              mask_zero=mask)

sequence_input_c = Input(shape=(padsize, MAX_WORD_LENGTH,), dtype='int32')

embedded_sequences_c = embedding_layer_c(sequence_input_c)

rone = Lambda(reshape_one)(embedded_sequences_c)

merge_m = 'sum'
merge_m_c = merge_m
dropout = 0.2
rec_dropout = dropout
gru_karakter = Bidirectional(
    GRU(CHAR_EMBEDDING_DIM, return_sequences=False, dropout=dropout, recurrent_dropout=rec_dropout), merge_mode=merge_m,
    weights=None)(rone)

rtwo = Lambda(reshape_two)(gru_karakter)

"""
Combine word + char model
"""
from keras.layers import Add, Subtract, Multiply, Average, Maximum

print "Model Choice:"
model_choice = 3  # input('Enter 1 for WE only, 2 for CE only, 3 for both: ')
merge_m = 'concat'
combine = 0
w_name_l = ''
w_name = ''
if model_choice == 1:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout),
                             merge_mode=merge_m, weights=None)(
        embedded_sequences)
elif model_choice == 2:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout),
                             merge_mode=merge_m, weights=None)(
        rtwo)
else:
    combine = 1
    if combine == 2:
        merge = Subtract()([embedded_sequences, rtwo])
    elif combine == 3:
        merge = Multiply()([embedded_sequences, rtwo])
    elif combine == 4:
        merge = Average()([embedded_sequences, rtwo])
    elif combine == 5:
        merge = Maximum()([embedded_sequences, rtwo])
    else:
        merge = Add()([embedded_sequences, rtwo])
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout),
                             merge_mode=merge_m, weights=None)(
        merge)

crf = CRF(len(label.index) + 1, learn_mode='marginal')(gru_kata)

preds = Dense(len(label.index) + 1, activation='softmax')(gru_kata)

print "Model Choice:"
model_choice = 1

model = Model(inputs=[sequence_input, sequence_input_c], outputs=[crf])
if model_choice == 2:
    model = Model(inputs=[sequence_input, sequence_input_c], outputs=[preds])

optimizer = 'adagrad'
loss = 'categorical_crossentropy'
model.summary()
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['acc'])

plot_model(model, to_file='model.png')

import pickle

load_m = 'y'
if 'y' in load_m:
    w_name = 'nw921'
    w_name_l = w_name
    load_c = 'y'
    m_layers_len = len(model.layers)
    if 'n' in load_c:
        m_layers_len -= 1
    for i in range(m_layers_len):
        with open(w_name + "-" + str(i) + ".wgt", "rb") as fp:
            w = pickle.load(fp)
            model.layers[i].set_weights(w)

epoch = 0
batch = 16
model.fit([np.array(x_train.padded), np.array(x_train_char)],
          [np.array(y_encoded)],
          epochs=epoch, batch_size=batch)

"""
Converting text data to int using index
"""
x_test_tmp1 = []
for sent in test.words:
    x_map = DM(sent, char.index, False)
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

x_test_char = []
for sent in x_test_tmp2:
    padded_sent = sent
    pad = padsize - len(sent)
    for i in range(pad):
        padded_sent = np.vstack((zeroes, padded_sent))
    x_test_char.append(padded_sent)

print('Padded until %s tokens.' % padsize)

"""
Evaluate
"""
x_test.pad(padsize)
y_test.pad(padsize)
results = []
print "Computing..."
raw_results = model.predict([np.array(x_test.padded), np.array(x_test_char)])
for raw_result in raw_results:
    result = []
    for token in raw_result:
        value = np.argmax(token)
        result.append(value)
    results.append(result)

"""
Sklearn evaluation
"""
label_index = range(1, len(label.index) + 1)
label_names = []
for key, value in sorted(label.index.iteritems(), key=lambda (k, v): (v, k)):
    label_names.append(key)

from sklearn.metrics import classification_report

# flatten list for sklearn evaluation
y_true = [item for sublist in y_test.padded for item in sublist]
y_pred = [item for sublist in results for item in sublist]
print "Sklearn evaluation:"
print classification_report(y_true, y_pred, labels=label_index, target_names=label_names)

from sklearn.metrics import f1_score

f1_mac = f1_score(y_true, y_pred, labels=label_index, average='macro')
f1_mic = f1_score(y_true, y_pred, labels=label_index, average='micro')
print 'F-1 Score:'
print max([f1_mac, f1_mic])

"""
Another test
"""
testname = raw_input('Enter test filename: ')
test = DL(testname)
x_test = DM(test.words, word.index)
y_test = DM(test.labels, label.index)
print "Number of OOV:", len(x_test.oov_index)
print "OOV word occurences:", x_test.oov

x_test_tmp1 = []
for sent in test.words:
    x_map = DM(sent, char.index, False)
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

x_test_char = []
for sent in x_test_tmp2:
    padded_sent = sent
    pad = padsize - len(sent)
    for i in range(pad):
        padded_sent = np.vstack((zeroes, padded_sent))
    x_test_char.append(padded_sent)

print('Padded until %s tokens.' % padsize)

x_test.pad(padsize)
y_test.pad(padsize)

results = []
print "Computing..."
raw_results = model.predict([np.array(x_test.padded), np.array(x_test_char)])
for raw_result in raw_results:
    result = []
    for token in raw_result:
        value = np.argmax(token)
        result.append(value)
    results.append(result)

"""
Sklearn evaluation
"""
label_index = range(1, len(label.index) + 1)
label_names = []
for key, value in sorted(label.index.iteritems(), key=lambda (k, v): (v, k)):
    label_names.append(key)

from sklearn.metrics import classification_report

# flatten list for sklearn evaluation
y_true = [item for sublist in y_test.padded for item in sublist]
y_pred = [item for sublist in results for item in sublist]
print "Sklearn evaluation:"
print classification_report(y_true, y_pred, labels=label_index, target_names=label_names)

from sklearn.metrics import f1_score

f1_mac = f1_score(y_true, y_pred, labels=label_index, average='macro')
f1_mic = f1_score(y_true, y_pred, labels=label_index, average='micro')
print 'F-1 Score:'
print max([f1_mac, f1_mic])
