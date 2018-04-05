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
from time import time
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

trainable = True  # word embedding is trainable or not
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
Preparing file
"""
train_name = sys.argv[1]
train = DL(train_name)
# percentage = input('Enter percentage of data to take: ')
percentage = 0.9
seed = sys.argv[2]
# seed = input('Enter seed for slicing data: ')
train.slice(percentage, seed)
test_name = sys.argv[1]
test = DL(test_name)
test.antislice(percentage, seed)

"""
Load pre-trained word embedding
"""

embeddings_index = {}
# WE_DIR = raw_input('Enter word embedding file name: ')
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
# CE_DIR = raw_input('Enter char embedding file name: ')
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

# char = DI(train.words + ce_words)
char = DI()
char.load('char')
# word = DI([train.words, [we_words]])
word = DI()
word.load('word')
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', word.cnt - 1, 'unique words.'
print 'Found', char.cnt - 1, 'unique chars.'
print 'Found', label.cnt - 1, 'unique labels.'

word.add([train.words])
print 'Found', word.cnt - 1, 'unique words.'

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

x_train = DM(train.words, word.index)
print "Number of train OOV:", len(x_train.oov_index)
print "OOV word occurences in train:", x_train.oov
x_test = DM(test.words, word.index)
print "Number of test OOV:", len(x_test.oov_index)
print "OOV word occurences: in test", x_test.oov

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
char_padsize = 24
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

# embeddingPrompt('word')
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

# embeddingPrompt('char')
embedding_layer_c = Embedding(len(char.index) + 1,
                              CHAR_EMBEDDING_DIM,
                              weights=[char_embedding_matrix],
                              input_length=MAX_WORD_LENGTH,
                              trainable=trainable,
                              mask_zero=mask)

sequence_input_c = Input(shape=(padsize, MAX_WORD_LENGTH,), dtype='int32')

embedded_sequences_c = embedding_layer_c(sequence_input_c)

rone = Lambda(reshape_one)(embedded_sequences_c)

merge_m = 'sum'  # raw_input('Enter merge mode for GRU Karakter: ')
dropout = 0.2  # input('Enter dropout for GRU: ')
rec_dropout = dropout  # input('Enter GRU Karakter recurrent dropout: ')
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
merge_m = 'concat'  # raw_input('Enter merge mode for GRU Kata: ')
# dropout = input('Enter GRU Karakter dropout: ')
# rec_dropout = input('Enter GRU Karakter recurrent dropout: ')
if model_choice == 1:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout),
                             merge_mode=merge_m, weights=None)(
        embedded_sequences)
elif model_choice == 2:
    gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout),
                             merge_mode=merge_m, weights=None)(
        rtwo)
else:
    combine = 1  # input('Enter 1 for Add, 2 for Subtract, 3 for Multiply, 4 for Average, 5 for Maximum: ')
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
model_choice = 1  # input('Enter 1 for CRF or 2 for Dense layer: ')

model = Model(inputs=[sequence_input, sequence_input_c], outputs=[crf])
if model_choice == 2:
    model = Model(inputs=[sequence_input, sequence_input_c], outputs=[preds])

optimizer = 'adagrad'  # raw_input('Enter optimizer (default rmsprop): ')
loss = 'categorical_crossentropy'  # raw_input('Enter loss function (default categorical_crossentropy): ')
model.summary()
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['acc'])

# plot_model(model, to_file='model.png')
"""
Converting testing text data to int using index
"""
x_test.pad(padsize)
y_test.pad(padsize)

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

label_index = range(1, len(label.index) + 1)
label_names = []
for key, value in sorted(label.index.iteritems(), key=lambda (k, v): (v, k)):
    label_names.append(key)

"""
Training
"""
epoch = 1
batch = int(sys.argv[4])
for ep in range(int(sys.argv[3])):
    print 'Epoch:', ep+1
    model.fit([np.array(x_train.padded), np.array(x_train_char)],
              [np.array(y_encoded)],
              epochs=epoch, batch_size=batch)
    results = []
    raw_results = model.predict([np.array(x_test.padded), np.array(x_test_char)])
    for raw_result in raw_results:
        result = []
        for token in raw_result:
            value = np.argmax(token)
            result.append(value)
        results.append(result)
    y_true = [item for sublist in y_test.padded for item in sublist]  # flatten list for sklearn evaluation
    y_pred = [item for sublist in results for item in sublist]
    print "Sklearn evaluation from epoch:", ep+1
    print classification_report(y_true, y_pred, labels=label_index, target_names=label_names)
    f1_mac = f1_score(y_true, y_pred, labels=label_index[1:], average='macro')
    f1_mic = f1_score(y_true, y_pred, labels=label_index[1:], average='micro')
    print 'F-1 Score (without O) from epoch:', ep+1
    print max([f1_mac, f1_mic])

"""
Evaluate
"""

"""
Sklearn evaluation
"""



"""
Save weight

save_m = raw_input('Do you want to save model weight? ')
if 'y' in save_m:
    w_name = raw_input('Enter file name to save weights: ')
    for i in range(len(model.layers)):
        with open(w_name+'-'+str(i)+'.wgt', 'wb') as fp:
            pickle.dump(model.layers[i].get_weights(), fp)

"""
# pm.predict('buah hati dia ingin memiliki cinta seorang anak tetapi aku tidak cinta kemudian menikah untuk kedua', padsize)