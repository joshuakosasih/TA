import ast
import csv
import sys
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from DataProcessor import DataIndexer as DI
from DataProcessor import DataLoader as DL
from DataProcessor import DataMapper as DM
from keras import Model
from keras import backend as K
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Concatenate
from keras.layers import Bidirectional, Dropout
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Lambda
from keras.utils import plot_model
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping

"""
Get Argvs
"""
a = []
for i in range(1, len(sys.argv)):
    a.append(sys.argv[i])

if a[0] == '.':
    trainable = True  # word embedding is trainable or not
else:
    trainable = ast.literal_eval(a[0])

mask = True  # mask pad (zeros) or not


def activationPrompt(name):
    print "List of Activation Functions\n" \
          "1. softmax\t\t2. elu\t\t3. selu\t\t4. softplus\t\t5. softsign\n" \
          "6. relu\t\t7. tanh\t\t8. sigmoid\t\t9. hard_sigmoid\t\t10. linear"
    choice = input('Enter type of activation function for ' + name + ': ')
    activations = ['softmax', 'elu', 'selu', 'softplus', 'softsign',
                   'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    return activations[choice - 1]


"""
Converting char text data to int using index
"""


def convertCharText2Int(dataload):
    x_tmp1 = []
    global char_padsize
    for sent in dataload.words:
        x_map = DM(sent, char.index, False)
        if x_map.padsize > char_padsize:
            char_padsize = x_map.padsize
        x_tmp1.append(x_map)

    x_tmp2 = []
    for sent in x_tmp1:
        sent.pad(char_padsize)
        x_tmp2.append(sent.padded)
    print('Padded until %s chars.' % char_padsize)
    zeroes = []
    for i in range(char_padsize):
        zeroes.append(0)
    x_char = []
    for sent in x_tmp2:
        padded_sent = sent
        pad = padsize - len(sent)
        for i in range(pad):
            padded_sent = np.vstack((zeroes, padded_sent))
        x_char.append(padded_sent)
    print('Padded until %s tokens.' % padsize)
    return x_char


"""
Preparing file
"""

train = DL('id-ud-train.pos')
test = DL('id-ud-dev.pos')
val = DL('id-ud-dev.pos')
# train.add('id-ud-dev.pos')

"""
Load pre-trained word embedding
"""

embeddings_index = {}
if a[1] == '.':
    WE_DIR = 'polyglot.vec'
else:
    WE_DIR = a[1]

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
if a[2] == '.':
    CE_DIR = 'polyglot-char.vec'
else:
    CE_DIR = a[2]

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
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', word.cnt - 1, 'unique words.'
print 'Found', char.cnt - 1, 'unique chars.'
print 'Found', label.cnt - 1, 'unique labels.'

"""
Create word embedding matrix
"""

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
Create char embedding matrix
"""

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
if a[3] == '.':
    trimlen = 188
else:
    trimlen = int(a[3])
train.trim(trimlen)
test.trim(trimlen)
val.trim(trimlen)

x_train = DM(train.words, word.index)
x_test = DM(test.words, word.index)
x_val = DM(val.words, word.index)
print "Number of OOV:", len(x_test.oov_index)
print "OOV word occurences:", x_test.oov
print "Number of OOV (val):", len(x_val.oov_index)
print "OOV word occurences (val):", x_val.oov
padsize = max([x_train.padsize, x_test.padsize])
x_train.pad(padsize)
x_test.pad(padsize)
x_val.pad(padsize)
print('Padded until %s tokens.' % padsize)

y_train = DM(train.labels, label.index)
y_test = DM(test.labels, label.index)
y_val = DM(val.labels, label.index)

y_train.pad(padsize)
y_test.pad(padsize)
y_val.pad(padsize)
y_encoded = to_categorical(y_train.padded)
y_val_enc = to_categorical(y_val.padded)

"""
Converting char text data to int using index
"""
char_padsize = 0
x_train_char = convertCharText2Int(train)
x_test_char = convertCharText2Int(test)
x_val_char = convertCharText2Int(val)
"""
Create keras word model
"""
MAX_SEQUENCE_LENGTH = padsize

# embeddingPrompt('word')
embedding_layer = Embedding(len(word.index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=trainable,
                            mask_zero=mask)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)
if a[4] == '.':
    drop = 0.5
else:
    drop = float(a[4])
dropout = Dropout(rate=drop)(embedded_sequences)

"""
Create keras char model
"""


def reshape_one(c):
    return K.reshape(c, (tf.shape(c)[0] * padsize, char_padsize, CHAR_EMBEDDING_DIM))


def reshape_two(c):
    if merge_m_c == 'concat':
        return K.reshape(c, (tf.shape(c)[0] / padsize, padsize, CHAR_EMBEDDING_DIM * 2))
    else:
        return K.reshape(c, (tf.shape(c)[0] / padsize, padsize, CHAR_EMBEDDING_DIM))


MAX_WORD_LENGTH = char_padsize

# embeddingPrompt('char')
embedding_layer_c = Embedding(len(char.index) + 1,
                              CHAR_EMBEDDING_DIM,
                              weights=[char_embedding_matrix],
                              input_length=MAX_WORD_LENGTH,
                              trainable=trainable,
                              mask_zero=mask)

sequence_input_c = Input(shape=(padsize, MAX_WORD_LENGTH,), dtype='int32')

embedded_sequences_c = embedding_layer_c(sequence_input_c)

dropout_c = Dropout(rate=drop)(embedded_sequences_c)

rone = Lambda(reshape_one)(dropout_c)
if a[5] == '.':
    merge_m = 'concat'
else:
    merge_m = a[5]
merge_m_c = merge_m
if a[6] == '.':
    dropout_gru = 0.0
else:
    dropout_gru = float(a[6])
rec_dropout = dropout_gru
gru_karakter = Bidirectional(
    GRU(CHAR_EMBEDDING_DIM, return_sequences=False, dropout=dropout_gru, recurrent_dropout=rec_dropout),
    merge_mode=merge_m, weights=None)(rone)

rtwo = Lambda(reshape_two)(gru_karakter)

"""
Combine word + char model
"""

print "Model Choice:"
if a[7] == '.':
    model_choice = 3
else:
    model_choice = int(a[7])
if a[8] == '.':
    merge_m = 'concat'
else:
    merge_m = a[8]
combine = 0
w_name_l = ''
w_name = ''
if model_choice == 1:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout_gru,
                                 recurrent_dropout=rec_dropout),
                             merge_mode=merge_m, weights=None)(
        dropout)
elif model_choice == 2:
    if merge_m_c == 'concat':
        gru_kata = Bidirectional(GRU(EMBEDDING_DIM * 2, return_sequences=True, dropout=dropout_gru,
                                     recurrent_dropout=rec_dropout), merge_mode=merge_m, weights=None)(rtwo)
    else:
        gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout_gru,
                                     recurrent_dropout=rec_dropout), merge_mode=merge_m, weights=None)(rtwo)
else:
    if merge_m_c == 'concat':
        merge = Concatenate()([dropout, rtwo])
        gru_kata = Bidirectional(GRU(EMBEDDING_DIM * 3, return_sequences=True, dropout=dropout_gru,
                                     recurrent_dropout=rec_dropout),
                                 merge_mode=merge_m, weights=None)(merge)
    else:
        if a[9] == '.':
            combine = 6
        else:
            combine = int(a[9])
        if combine == 2:
            merge = Subtract()([dropout, rtwo])
        elif combine == 3:
            merge = Multiply()([dropout, rtwo])
        elif combine == 4:
            merge = Average()([dropout, rtwo])
        elif combine == 5:
            merge = Maximum()([dropout, rtwo])
        elif combine == 6:
            merge = Concatenate()([dropout, rtwo])
        else:
            merge = Add()([dropout, rtwo])
        if combine == 6:
            gru_kata = Bidirectional(GRU(EMBEDDING_DIM * 2, return_sequences=True, dropout=dropout_gru,
                                         recurrent_dropout=rec_dropout),
                                     merge_mode=merge_m, weights=None)(merge)
        else:
            gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout_gru,
                                         recurrent_dropout=rec_dropout),
                                     merge_mode=merge_m, weights=None)(merge)

crf = CRF(len(label.index) + 1, learn_mode='marginal')(gru_kata)

model = Model(inputs=[sequence_input, sequence_input_c], outputs=[crf])

if a[10] == '.':
    optimizer = 'rmsprop'
else:
    optimizer = a[10]
if a[11] == '.':
    loss = 'categorical_crossentropy'
else:
    loss = a[11]
model.summary()
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['acc'])

# plot_model(model, to_file='model.png')

if a[12] != '.':
    epoch = a[12]
if a[13] == '.':
    batch = 32
else:
    batch = int(a[13])
if a[14] == '.':
    use_val = 'y'
else:
    use_val = a[14]
if 'y' in use_val:
    val_data = ([np.array(x_val.padded), np.array(x_val_char)], [np.array(y_val_enc)])
else:
    val_data = None
if a[12] == '.':
    use_estop = 'y'
else:
    use_estop = 'n'
callback = None
if 'y' in use_estop:
    epoch = 30
    callback = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
model.fit([np.array(x_train.padded), np.array(x_train_char)],
          [np.array(y_encoded)], validation_data=val_data, validation_split=0.1,
          epochs=epoch, batch_size=batch, callbacks=[callback])

"""
Evaluate
"""

mateval = []
for labr in range(label.cnt):
    row = []
    for labc in range(label.cnt):
        row.append(0)
    mateval.append(row)

results = []
print "Computing..."
raw_results = model.predict([np.array(x_test.padded), np.array(x_test_char)])
for raw_result in raw_results:
    result = []
    for token in raw_result:
        value = np.argmax(token)
        result.append(value)
    results.append(result)

total_nonzero = 0  # to get labelled token total number
for i, sent in enumerate(y_test.padded):
    for j, token in enumerate(sent):
        pred = results[i][j]
        answ = token
        mateval[answ][pred] += 1  # row shows label and column shows prediction given
        if not answ == 0:
            total_nonzero += 1

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
f1 = max([f1_mac, f1_mic])
print 'F-1 Score:'
print f1

"""
Save weight
"""
rnow = datetime.now()

save_m = 'y'
if 'y' in save_m:
    w_name = str(rnow.date())[5:] + '_' \
             + str(rnow.time())[:-10] + '_' \
             + str(round(f1, 3) * 1000)[:-2]
    for i in range(len(model.layers)):
        with open(w_name + '_' + str(i) + '.wgt', 'wb') as fp:
            pickle.dump(model.layers[i].get_weights(), fp)

"""
Logging Experiment
"""
if 'y' in use_estop:
    epoch = callback.stopped_epoch

logcsv = open('log.csv', 'a')
writer = csv.writer(logcsv, delimiter=',')
load_m = 'n'
writer.writerow(
    ['no', str(rnow.date()), str(rnow.time())[:-10], train.filename, test.filename, WE_DIR, CE_DIR,
     word.cnt - 1, char.cnt - 1, len(x_test.oov_index), padsize, char_padsize, trainable, merge_m_c,
     merge_m, drop, dropout_gru, model_choice, combine, optimizer, loss, load_m, w_name_l, epoch, batch, f1_mac,
     f1_mic, save_m, w_name])

logcsv.close()
