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
from keras.utils import plot_model
from keras.utils import to_categorical
from keras_contrib.layers import CRF

"""
Preparing file
"""

train = DL('id-ud-train.pos')
test = DL('id-ud-test.pos')

"""
Create Word & Label Index
"""

word = DI([train.words, test.words])
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', word.cnt - 1, 'unique words.'
print 'Found', label.cnt - 1, 'unique labels.'

"""
Load pre-trained embedding
"""

embeddings_index = {}
GLOVE_DIR = 'WE_w.txt'

f = open(GLOVE_DIR, 'r')
for line in f:
    values = line.split()
    wrd = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[wrd] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 200

notfound = []  # list kata yang tidak terdapat dalam embedding
embedding_matrix = np.zeros((len(word.index) + 1, int(EMBEDDING_DIM)))
for wrd, i in word.index.items():
    embedding_vector = embeddings_index.get(wrd)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        notfound.append(wrd)

print('%s words not found in embedding.' % len(notfound))

"""
Converting text data to int using index
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
Create keras model
"""

MAX_SEQUENCE_LENGTH = padsize

embedding_layer = Embedding(len(word.index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True), merge_mode='concat', weights=None)(
    embedded_sequences)

crf = CRF(len(label.index) + 1, learn_mode='marginal')(gru_kata)

preds = Dense(len(label.index) + 1, activation='softmax')(gru_kata)

model = Model(sequence_input, crf)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

plot_model(model, to_file='model.png')

model.fit(np.array(x_train.padded), np.array(y_encoded), epochs=2, batch_size=128)

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
raw_results = model.predict(np.array(x_test.padded))
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

"""
Predict function
"""

# pm.predict('buah hati dia ingin memiliki cinta seorang anak tetapi aku tidak cinta kemudian menikah untuk kedua', padsize)
