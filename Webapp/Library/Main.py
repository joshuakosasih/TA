import csv
import json
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from DataProcessor import DataLoader as DL
from DataProcessor import DataIndexer as DI
from DataProcessor import DataMapper as DM
from DataProcessor import DataPreprocessor as DP
from boto.dynamodb2 import results
from keras import Model
from keras import backend as K
from keras.layers import Bidirectional, Dropout
from keras.layers import Concatenate
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


class SeqTagger:
    mask = True  # mask pad (zeros) or not
    EMBEDDING_DIM = 64
    CHAR_EMBEDDING_DIM = 64
    padsize = 188
    char_padsize = 41
    dropout_embedding = 0.4
    dropout_gru = 0.5
    optimizer = 'adagrad'
    loss = 'poisson'
    patience = 3

    def __init__(self):
        self.coefs = []
        self.textinput = ''
        self.train = ''
        self.val = ''
        self.test = ''
        self.char = ''
        self.word = ''
        self.label = ''
        self.x_train = ''
        self.x_val = ''
        self.x_test = ''
        self.x_test_char = ''
        self.y_train = ''
        self.y_test = ''
        self.y_val = ''
        self.y_encoded = ''
        self.y_val_enc = ''
        self.x_test_char = ''
        self.x_train_char = ''
        self.x_val_char = ''
        self.model = {}
        self.results = []
        self.data = {}
        self.json_data = {}

    def createModel(self, traindata, valdata, testdata, wordemb, charemb):
        self.train = DL(traindata)
        self.val = DL(valdata)
        self.test = DL(testdata)

        # Load pre-trained embedding
        embeddings_index, we_words = self.pretrainedEmbeddingLoader(wordemb)
        char_embeddings_index, ce_words = self.pretrainedEmbeddingLoader(charemb)

        # Create Word & Label Index
        self.char = DI(self.train.words + ce_words)
        self.word = DI([self.train.words, [we_words]])
        self.label = DI([self.train.labels])
        print 'Found', self.word.cnt - 1, 'unique words.'
        print 'Found', self.char.cnt - 1, 'unique chars.'
        print 'Found', self.label.cnt - 1, 'unique labels.'

        # Create word embedding matrix
        self.EMBEDDING_DIM = len(self.coefs)
        embedding_matrix = np.zeros((len(self.word.index) + 1, int(self.EMBEDDING_DIM)))
        for wrd, i in self.word.index.items():
            embedding_vector = embeddings_index.get(wrd)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Create char embedding matrix
        char_embedding_matrix = np.zeros((len(self.char.index) + 1, int(self.EMBEDDING_DIM)))
        for chars, i in self.char.index.items():
            embedding_vector = char_embeddings_index.get(chars)
            if embedding_vector is not None:
                char_embedding_matrix[i] = embedding_vector

        trimlen = self.padsize
        self.train.trim(trimlen)
        self.test.trim(trimlen)
        self.val.trim(trimlen)

        self.x_train = DM(self.train.words, self.word.index)
        self.x_test = DM(self.test.words, self.word.index)
        self.x_val = DM(self.val.words, self.word.index)
        print "Number of OOV:", len(self.x_test.oov_index)
        print "OOV word occurences:", self.x_test.oov
        print "Number of OOV (val):", len(self.x_val.oov_index)
        print "OOV word occurences (val):", self.x_val.oov
        padsize = self.padsize
        self.x_train.pad(padsize)
        self.x_test.pad(padsize)
        self.x_val.pad(padsize)
        print('Padded until %s tokens.' % padsize)

        self.y_train = DM(self.train.labels, self.label.index)
        self.y_test = DM(self.test.labels, self.label.index)
        self.y_val = DM(self.val.labels, self.label.index)

        self.y_train.pad(padsize)
        self.y_test.pad(padsize)
        self.y_val.pad(padsize)
        self.y_encoded = to_categorical(self.y_train.padded)
        self.y_val_enc = to_categorical(self.y_val.padded)

        # Converting char text data to int using index
        self.x_test_char = self.convertCharText2Int(self.test)
        self.x_train_char = self.convertCharText2Int(self.train)
        self.x_val_char = self.convertCharText2Int(self.val)

        # Create keras word model
        MAX_SEQUENCE_LENGTH = self.padsize
        embedding_layer = Embedding(len(self.word.index) + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    mask_zero=self.mask)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        embedded_sequences = embedding_layer(sequence_input)
        drop = self.dropout_embedding
        dropout = Dropout(rate=drop)(embedded_sequences)

        # Create keras char model
        def reshape_one(c):
            return K.reshape(c, (tf.shape(c)[0] * self.padsize, self.char_padsize, self.CHAR_EMBEDDING_DIM))

        def reshape_two(c):
            if merge_m_c == 'concat':
                return K.reshape(c, (tf.shape(c)[0] / self.padsize, self.padsize, self.CHAR_EMBEDDING_DIM * 2))
            else:
                return K.reshape(c, (tf.shape(c)[0] / self.padsize, self.padsize, self.CHAR_EMBEDDING_DIM))

        MAX_WORD_LENGTH = self.char_padsize

        embedding_layer_c = Embedding(len(self.char.index) + 1,
                                      self.EMBEDDING_DIM,
                                      weights=[char_embedding_matrix],
                                      input_length=MAX_WORD_LENGTH,
                                      mask_zero=self.mask)

        sequence_input_c = Input(shape=(self.padsize, MAX_WORD_LENGTH,), dtype='int32')
        embedded_sequences_c = embedding_layer_c(sequence_input_c)
        dropout_c = Dropout(rate=drop)(embedded_sequences_c)

        rone = Lambda(reshape_one)(dropout_c)
        merge_m = 'concat'
        merge_m_c = merge_m
        dropout_gru = self.dropout_gru
        rec_dropout = dropout_gru
        gru_karakter = Bidirectional(
            GRU(self.CHAR_EMBEDDING_DIM, return_sequences=False, dropout=dropout_gru, recurrent_dropout=rec_dropout),
            merge_mode=merge_m, weights=None)(rone)

        rtwo = Lambda(reshape_two)(gru_karakter)

        # Combine word + char model
        merge_m = 'concat'
        merge = Concatenate()([dropout, rtwo])
        gru_kata = Bidirectional(GRU(self.EMBEDDING_DIM * 3, return_sequences=True, dropout=dropout_gru,
                                     recurrent_dropout=rec_dropout),
                                 merge_mode=merge_m, weights=None)(merge)

        crf = CRF(len(self.label.index) + 1, learn_mode='marginal')(gru_kata)
        self.model = Model(inputs=[sequence_input, sequence_input_c], outputs=[crf])

        optimizer = self.optimizer
        loss = self.loss
        self.model.summary()
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['acc'])

    def trainFit(self):
        val_data = ([np.array(self.x_val.padded), np.array(self.x_val_char)], [np.array(self.y_val_enc)])
        callback = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0, mode='auto')
        self.model.fit([np.array(self.x_train.padded), np.array(self.x_train_char)],
                       [np.array(self.y_encoded)], validation_data=val_data, validation_split=0.1,
                       epochs=100, batch_size=32, callbacks=[callback])

    def evaluate(self):
        self.results = []
        print "Computing..."
        raw_results = self.model.predict([np.array(self.x_test.padded), np.array(self.x_test_char)])
        for raw_result in raw_results:
            result = []
            for token in raw_result:
                value = np.argmax(token)
                result.append(value)
            self.results.append(result)

        label_index = range(1, len(self.label.index) + 1)
        label_names = []
        for key, value in sorted(self.label.index.iteritems(), key=lambda (k, v): (v, k)):
            label_names.append(key)

        # flatten list for sklearn evaluation
        y_true = [item for sublist in self.y_test.padded for item in sublist]
        y_pred = [item for sublist in self.results for item in sublist]
        print "Sklearn evaluation:"
        print classification_report(y_true, y_pred, labels=label_index, target_names=label_names)

        f1_mac = f1_score(y_true, y_pred, labels=label_index[1:], average='macro')
        f1_mic = f1_score(y_true, y_pred, labels=label_index[1:], average='micro')
        f1 = max([f1_mac, f1_mic])
        print 'F-1 Score:'
        print f1

    def predict(self, text):
        self.textinput = text
        self.test = DP(text)
        self.x_test = DM(self.test.words, self.word.index)
        print "Number of OOV:", len(self.x_test.oov_index)
        print "OOV word occurences:", self.x_test.oov

        self.x_test.pad(self.padsize)
        print('Padded until %s tokens.' % self.padsize)

        self.x_test_char = self.convertCharText2Int(self.test)

        self.results = []
        print "Computing..."
        print self.x_test.padded
        print self.x_test_char
        raw_results = self.model.predict([np.array(self.x_test.padded), np.array(self.x_test_char)])
        for raw_result in raw_results:
            result = []
            for token in raw_result:
                value = np.argmax(token)
                result.append(value)
            self.results.append(result)

        temp = self.results[0]
        li = self.label.index
        keys = li.keys()
        values = li.values()
        self.results = []
        start = False
        for token in temp:
            if token != 0:
                start = True
            if start:
                if token == 0:
                    self.results.append('?')
                else:
                    self.results.append(keys[values.index(token)])

        print self.test.words[0]
        print self.results

        self.data = {'words': self.test.words[0], 'labels': self.results}
        self.json_data = json.dumps(self.data)
        return self.json_data

    def log(self):
        self.textoutput = ''
        for token in self.results:
            self.textoutput = self.textoutput + token + ' '
        rnow = datetime.now()
        logcsv = open('log.csv', 'a')
        writer = csv.writer(logcsv, delimiter=',')
        writer.writerow(
            ['no', str(rnow.date()), str(rnow.time())[:-10], self.w_name, self.word.cnt - 1,
             self.char.cnt - 1, self.textinput, len(self.x_test.oov_index), self.textoutput])
        logcsv.close()

    def convertCharText2Int(self, dataload):
        x_tmp1 = []
        for sent in dataload.words:
            x_map = DM(sent, self.char.index, False)
            if x_map.padsize > self.char_padsize:
                self.char_padsize = x_map.padsize
            x_tmp1.append(x_map)

        x_tmp2 = []
        for sent in x_tmp1:
            sent.pad(self.char_padsize)
            x_tmp2.append(sent.padded)
        print('Padded until %s chars.' % self.char_padsize)
        zeroes = []
        for i in range(self.char_padsize):
            zeroes.append(0)
        x_char = []
        for sent in x_tmp2:
            padded_sent = sent
            pad = self.padsize - len(sent)
            for i in range(pad):
                padded_sent = np.vstack((zeroes, padded_sent))
            x_char.append(padded_sent)
        print('Padded until %s tokens.' % self.padsize)
        return x_char

    def pretrainedEmbeddingLoader(self, filename):
        embeddings_index = {}
        f = open(filename, 'r')
        for line in f:
            values = line.split()
            token = values[0]
            self.coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[token] = self.coefs
        f.close()
        embs = []
        for token in embeddings_index:
            embs.append(token)
        return embeddings_index, embs

    def saveModel(self, w_name):
        for i in range(len(self.model.layers)):
            with open(w_name + '_' + str(i) + '.wgt', 'wb') as fp:
                pickle.dump(self.model.layers[i].get_weights(), fp)

    def loadModel(self, w_name):
        m_layers_len = len(self.model.layers)
        for i in range(m_layers_len):
            with open(w_name + '_' + str(i) + ".wgt", "rb") as fp:
                w = pickle.load(fp)
                self.model.layers[i].set_weights(w)