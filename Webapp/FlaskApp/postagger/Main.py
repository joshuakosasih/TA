import csv
import json
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
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
from keras_contrib.layers import CRF

class POSTagger:
    mask = True  # mask pad (zeros) or not
    EMBEDDING_DIM = 64
    CHAR_EMBEDDING_DIM = 64
    padsize = 188
    char_padsize = 25

    def __init__(self):
        self.textinput = ''
        self.test = ''
        self.x_test = ''
        self.x_test_char = ''
        self.results = []
        self.data = {}
        self.json_data = {}

        self.char = DI()
        self.char.load('char')
        self.word = DI()
        self.word.load('word')
        self.label = DI()
        self.label.load('label')

        print 'Found', self.word.cnt - 1, 'unique words.'
        print 'Found', self.char.cnt - 1, 'unique chars.'
        print 'Found', self.label.cnt - 1, 'unique labels.'

        embedding_matrix = np.zeros((len(self.word.index) + 1, int(self.EMBEDDING_DIM)))
        char_embedding_matrix = np.zeros((len(self.char.index) + 1, int(self.CHAR_EMBEDDING_DIM)))

        """
        Create keras word model
        """

        MAX_SEQUENCE_LENGTH = self.padsize
        embedding_layer = Embedding(len(self.word.index) + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    mask_zero=self.mask)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        embedded_sequences = embedding_layer(sequence_input)
        drop = 0.4
        dropout = Dropout(rate=drop)(embedded_sequences)

        """
        Create keras char model
        """

        def reshape_one(c):
            return K.reshape(c, (tf.shape(c)[0] * self.padsize, self.char_padsize, self.CHAR_EMBEDDING_DIM))

        def reshape_two(c):
            if merge_m_c == 'concat':
                return K.reshape(c, (tf.shape(c)[0] / self.padsize, self.padsize, self.CHAR_EMBEDDING_DIM * 2))
            else:
                return K.reshape(c, (tf.shape(c)[0] / self.padsize, self.padsize, self.CHAR_EMBEDDING_DIM))

        MAX_WORD_LENGTH = self.char_padsize

        # embeddingPrompt('char')
        embedding_layer_c = Embedding(len(self.char.index) + 1,
                                      self.CHAR_EMBEDDING_DIM,
                                      weights=[char_embedding_matrix],
                                      input_length=MAX_WORD_LENGTH,
                                      mask_zero=self.mask)

        sequence_input_c = Input(shape=(self.padsize, MAX_WORD_LENGTH,), dtype='int32')

        embedded_sequences_c = embedding_layer_c(sequence_input_c)

        dropout_c = Dropout(rate=drop)(embedded_sequences_c)

        rone = Lambda(reshape_one)(dropout_c)
        merge_m = 'concat'
        merge_m_c = merge_m
        dropout_gru = 0.5
        rec_dropout = dropout_gru
        gru_karakter = Bidirectional(
            GRU(self.CHAR_EMBEDDING_DIM, return_sequences=False, dropout=dropout_gru, recurrent_dropout=rec_dropout),
            merge_mode=merge_m, weights=None)(rone)

        rtwo = Lambda(reshape_two)(gru_karakter)

        """
        Combine word + char model
        """

        merge_m = 'concat'
        merge = Concatenate()([dropout, rtwo])
        gru_kata = Bidirectional(GRU(self.EMBEDDING_DIM * 3, return_sequences=True, dropout=dropout_gru,
                                     recurrent_dropout=rec_dropout),
                                 merge_mode=merge_m, weights=None)(merge)

        crf = CRF(len(self.label.index) + 1, learn_mode='marginal')(gru_kata)

        self.model = Model(inputs=[sequence_input, sequence_input_c], outputs=[crf])

        optimizer = 'adagrad'
        loss = 'poisson'
        self.model.summary()
        self.model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['acc'])

        self.w_name = '05-17_13:37_921'
        m_layers_len = len(self.model.layers)
        for i in range(m_layers_len):
            with open(self.w_name + "_" + str(i) + ".wgt", "rb") as fp:
                w = pickle.load(fp)
                self.model.layers[i].set_weights(w)
        
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
