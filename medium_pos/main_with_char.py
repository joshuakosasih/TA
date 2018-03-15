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
import tensorflow as tf

trainable = False
mask = False


def embeddingPrompt(name):
    global trainable
    trainable = raw_input('Is ' + name + ' embedding trainable? ')
    global mask
    mask = raw_input('Use mask for zeros in ' + name + ' embedding? ')
    if 'y' in trainable:
        trainable = True
    else:
        trainable = False
    if 'y' in mask:
        mask = True
    else:
        mask = False


"""
Preparing file
"""

train = DL('id-ud-train')
test = DL('id-ud-test')

"""
Create Word & Label Index
"""

char = DI(train.words + test.words)
word = DI([train.words, test.words])
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', word.cnt - 1, 'unique words.'
print 'Found', char.cnt - 1, 'unique chars.'
print 'Found', label.cnt - 1, 'unique labels.'

"""
Load pre-trained word embedding
"""

embeddings_index = {}
WE_DIR = raw_input('Enter word embedding file name: ')
# WE_DIR = 'polyglot.txt'

print 'Loading', WE_DIR, '...'
f = open(WE_DIR, 'r')
for line in f:
    values = line.split()
    wrd = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[wrd] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = len(coefs)

notfound = []  # list kata yang tidak terdapat dalam embedding
embedding_matrix = np.zeros((len(word.index) + 1, int(EMBEDDING_DIM)))
for wrd, i in word.index.items():
    embedding_vector = embeddings_index.get(wrd)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        notfound.append(wrd)

print('%s unique words not found in embedding.' % len(notfound))

"""
Load pre-trained char embedding
"""

char_embeddings_index = {}
CE_DIR = raw_input('Enter char embedding file name: ')
# CE_DIR = 'polyglot-char.txt'

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
Converting word text data to int using index
"""

x_train = DM(train.words, word.index)
x_test = DM(test.words, word.index)

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

# char_padsize = max([x_train.char_padsize, x_test.char_padsize])

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

embeddingPrompt('word')
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

embeddingPrompt('char')
embedding_layer_c = Embedding(len(char.index) + 1,
                              CHAR_EMBEDDING_DIM,
                              weights=[char_embedding_matrix],
                              input_length=MAX_WORD_LENGTH,
                              trainable=trainable,
                              mask_zero=mask)

sequence_input_c = Input(shape=(padsize, MAX_WORD_LENGTH,), dtype='int32')

embedded_sequences_c = embedding_layer_c(sequence_input_c)

rone = Lambda(reshape_one)(embedded_sequences_c)

merge_m = raw_input('Enter merge mode for GRU Karakter: ')
gru_karakter = Bidirectional(GRU(CHAR_EMBEDDING_DIM, return_sequences=False), merge_mode=merge_m, weights=None)(rone)

rtwo = Lambda(reshape_two)(gru_karakter)

"""
Combine word + char model
"""
from keras.layers import Add

print "Model Choice:"
model_choice = input('Enter 1 for WE only, 2 for CE only, 3 for both: ')
merge_m = raw_input('Enter merge mode for GRU Kata: ')

if model_choice == 1:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True), merge_mode=merge_m, weights=None)(
        embedded_sequences)
elif model_choice == 2:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True), merge_mode=merge_m, weights=None)(
        rtwo)
else:
    merge = Add()([embedded_sequences, rtwo])
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True), merge_mode=merge_m, weights=None)(
        merge)

crf = CRF(len(label.index) + 1, learn_mode='marginal')(gru_kata)

preds = Dense(len(label.index) + 1, activation='softmax')(gru_kata)

print "Model Choice:"
model_choice = input('Enter 1 for CRF or 2 for Dense layer: ')

model = Model(inputs=[sequence_input, sequence_input_c], outputs=[crf])
if model_choice == 2:
    model = Model(inputs=[sequence_input, sequence_input_c], outputs=[preds])

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

plot_model(model, to_file='model.png')

epoch = input('Enter number of epochs: ')
batch = input('Enter number of batch size: ')
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

mateval = []
for labr in range(label.cnt):
    row = []
    for labc in range(label.cnt):
        row.append(0)
    mateval.append(row)

x_test.pad(padsize)
results = []
print "Computing..."
raw_results = model.predict([np.array(x_test.padded), np.array(x_test_char)])
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

print "Manual evaluation: (didn't understand why I made this)"
print "True", total_true
print "False", total_false
print "True percentage", float(total_true) / float(total_nonzero)

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
Predict function
"""

# pm.predict('buah hati dia ingin memiliki cinta seorang anak tetapi aku tidak cinta kemudian menikah untuk kedua', padsize)
