import os
import nltk
import numpy as np
from nltk.tag import StanfordNERTagger
from keras import Model
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from DataProcessor import DataLoader as DL
from DataProcessor import DataIndexer as DI
from DataProcessor import DataMapper as DM

"""
Preparing file
"""

train = DL('id-ud-train')
test = DL('id-ud-test')

"""
Create Word & Label Index
"""

word = DI([train.words, test.words])
label = DI([train.labels])  # training label and testing label should be the same

print 'Found', word.cnt - 1, 'unique words.'
print 'Found', label.cnt-1, 'unique labels.'

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

gru_kata = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True), merge_mode='concat', weights=None)(embedded_sequences)

crf = CRF(len(label.index)+1, learn_mode='marginal')(gru_kata)

preds = Dense(len(label.index)+1, activation='softmax')(gru_kata)

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
for labr in labels_index:
    row = []
    for labc in labels_index:
        row.append(0)
    mateval.append(row)

results = []
raw_results = model.predict(x_padded[:5]) # change x_padded
for raw_result in raw_results:
    result = []
    for token in raw_result:
        value = np.argmax(token)
        result.append(value)
    results.append(result)

total_nonzero = 0 # to get labelled token total number
for i, sent in enumerate(y_padded_2):
    for j, token in enumerate(sent):
        pred = results[i][j]
        answ = token
        mateval[answ][pred] = mateval[answ][pred] + 1 # row shows label and column shows prediction given
        if not answ == 0:
            total_nonzero = total_nonzero + 1

total_true = 0
for i in range(1, len(mateval)):
    total_true = total_true + mateval[i][i]

total_false = total_nonzero - total_true

"""
Predict function
"""

def predict(sentence):
    sent = nltk.word_tokenize(sentence) #tokenize
    se = []
    for it in range(len(sent), MAX_SEQUENCE_LENGTH): #padding
        se.append(0)
    
    for token in sent: #indexing
        se.append(word_index[token])
    
    se = np.array([se]) #change to np array
    
    result = model.predict(se)[0] #get prediction result
    res = []
    for token in result:
        value = np.argmax(token)
        if value == 0:
            res.append('~')
        else:
            key = labels_index.keys()[labels_index.values().index(value)]
            res.append(key)
    
    return res
    print res

predict('buah hati dia ingin memiliki cinta seorang anak tetapi aku tidak cinta kemudian menikah untuk kedua')


