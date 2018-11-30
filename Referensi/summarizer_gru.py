#%%

# Define padding constants here
MAX_SENT_LEN = 30
MAX_DOC_LEN = 100
MAX_CLUSTER_LEN = 10

# Define architecture constants here
EMBEDDING_LEN = 300

GRU_SENT_UNITS = 300
GRU_SENT_DROPOUT = 0.2
GRU_SENT_R_DROPOUT = 0.2

GRU_DOC_UNITS = 300
GRU_DOC_DROPOUT = 0.2
GRU_DOC_R_DROPOUT = 0.2

GCN_UNITS = 300
GCN_DROPOUT = 0.5

SALIENCE_UNIT = 512
EPOCHS = 20

# Should punctuation be one unique word? (for tokenizer and embedding)
UNIQUE_PUNCT = False

# Should always be 1
BATCH_SIZE = 1

# For graph normalization symmetry
SYM_NORM = True
            
#%%

import preprocess

directory = 'indonesian'

# Prepare train
docs_train_dir = '{}/train/docs'.format(directory)
docs_text_train, docs_documents_train = preprocess.prepare_text(docs_train_dir, UNIQUE_PUNCT)

sums_train_dir = '{}/train/summaries'.format(directory)
sums_text_train, sums_documents_train = preprocess.prepare_text(sums_train_dir, UNIQUE_PUNCT)

#%%

# Prepare validation
docs_val_dir = '{}/validation/docs'.format(directory)
docs_text_val, docs_documents_val = preprocess.prepare_text(docs_val_dir, UNIQUE_PUNCT)

sums_val_dir = '{}/validation/summaries'.format(directory)
sums_text_val, sums_documents_val = preprocess.prepare_text(sums_val_dir, UNIQUE_PUNCT)

#%%

# Prepare test
docs_test_dir = '{}/test/docs'.format(directory)
docs_text_test, docs_documents_test = preprocess.prepare_text(docs_test_dir, UNIQUE_PUNCT)

sums_test_dir = '{}/test/summaries'.format(directory)
sums_text_test, sums_documents_test = preprocess.prepare_text(sums_test_dir, UNIQUE_PUNCT)

#%%

# Transform sentence into sequence
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(filters='\n\t')
tokenizer_input = docs_documents_train + sums_documents_train
tokenizer.fit_on_texts(tokenizer_input)

vector_train = preprocess.vectorize(docs_text_train, tokenizer, MAX_SENT_LEN)
vector_test = preprocess.vectorize(docs_text_test, tokenizer, MAX_SENT_LEN)
vector_val = preprocess.vectorize(docs_text_val, tokenizer, MAX_SENT_LEN)

#%%
    
import numpy as np

def pad_documents(docs):
    for i, doc in enumerate(docs):
        pad_len = MAX_DOC_LEN - len(doc)
        if pad_len < 0:
            docs[i] = doc[:MAX_DOC_LEN]
        else:
            docs[i] = np.concatenate((doc, np.zeros((pad_len, MAX_SENT_LEN))))

def pad_clusters(clusters):
    for k, v in clusters.items():
        pad_len = MAX_CLUSTER_LEN - len(v)
        if pad_len < 0:
            clusters[k] = v[:MAX_CLUSTER_LEN]
        else:
            for i in range(pad_len):
                v.append(np.zeros((MAX_DOC_LEN, MAX_SENT_LEN)))

def transform_to_data(news_vector):
    data_list = []
    for k, v in news_vector.items():
        data_list.append(np.array(v, dtype=np.int32))
    return np.array(data_list, dtype=np.int32)

for k, v in vector_train.items():
    pad_documents(v)
    pad_clusters(vector_train)
    
for k, v in vector_test.items():
    pad_documents(v)
    pad_clusters(vector_test)
    
for k, v in vector_val.items():
    pad_documents(v)
    pad_clusters(vector_val)

X_train = transform_to_data(vector_train)
X_test = transform_to_data(vector_test)
X_val = transform_to_data(vector_val)

#%%

from preprocess import create_target

y_train = create_target(docs_text_train, sums_text_train, MAX_DOC_LEN, MAX_CLUSTER_LEN)
y_test = create_target(docs_text_test, sums_text_test, MAX_DOC_LEN, MAX_CLUSTER_LEN)
y_val = create_target(docs_text_val, sums_text_val, MAX_DOC_LEN, MAX_CLUSTER_LEN)

#%%

from keras import backend as K
import tensorflow as tf

def reshape_to_sentence(c):
    return K.reshape(c, (tf.shape(c)[0] * MAX_CLUSTER_LEN * MAX_DOC_LEN, MAX_SENT_LEN))

def reshape_to_document(c):
    return K.reshape(c, (tf.shape(c)[0] // MAX_DOC_LEN, MAX_DOC_LEN, GRU_DOC_UNITS))

def cluster_embedding(c):
    return K.mean(c, axis=0, keepdims=True)

def repeat_cluster(c):
    return K.repeat_elements(c, MAX_CLUSTER_LEN * MAX_DOC_LEN, axis=0)

def last_reshape(c):
    return K.reshape(c, (tf.shape(c)[0] // (MAX_CLUSTER_LEN * MAX_DOC_LEN), MAX_CLUSTER_LEN * MAX_DOC_LEN))

#%%

# Prepare Embedding weights
embedding_filename = 'embeddings/glove.6B.300d.txt'
embeddings_index = {}
f = open(embedding_filename, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_LEN))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#%%

from keras.layers import Dense, Input, Lambda, GRU, Embedding, Activation, add
from keras.models import Model

# Input layer
main_input = Input(shape=(MAX_CLUSTER_LEN, MAX_DOC_LEN, MAX_SENT_LEN), name='main_input')
c_to_s = Lambda(reshape_to_sentence, name='sentences')(main_input)

# Word Embedding
embedding = Embedding(output_dim=EMBEDDING_LEN,
                      input_dim=len(tokenizer.word_index)+1,
                      weights=[embedding_matrix],
                      input_length=MAX_SENT_LEN,
                      name='word_embedding')(c_to_s)

# GRU Sentence
gru_sent = GRU(units=GRU_SENT_UNITS, dropout=GRU_SENT_DROPOUT, recurrent_dropout=GRU_SENT_R_DROPOUT, name='gru_sentence')(embedding)

# Sentence to Document
s_to_d = Lambda(reshape_to_document, name='documents')(gru_sent)

# Combine GRU Sentence and GRU Cluster
gru_doc = GRU(units=GRU_DOC_UNITS, dropout=GRU_DOC_DROPOUT, recurrent_dropout=GRU_DOC_R_DROPOUT, name='gru_document')(s_to_d)

# Cluster Embedding
cls_embedding = Lambda(cluster_embedding, name='cluster_embedding')(gru_doc)

# Sentence Embedding Dense
s_dense = Dense(SALIENCE_UNIT, activation='linear', name='sentence_dense')(gru_sent)

# Cluster Embedding Dense
c_dense = Dense(SALIENCE_UNIT, activation='linear', name='cluster_dense')(cls_embedding)

# Repeat Cluster
rp_cluster = Lambda(repeat_cluster, name='repeat_cluster')(c_dense)

# Cluster Embedding Dense
combined = add([s_dense, rp_cluster])

# Salience Score
salience = Activation('tanh', name='tanh')(combined)

salience_x = Dense(1, activation='linear', name='sakience')(salience)

last = Lambda(last_reshape)(salience_x)

last_softmax = Activation('softmax')(last)

# Model
model = Model(inputs=[main_input], outputs=[last_softmax])

model.summary()

#%%

# For sample
def square_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Real loss function
def custom_loss(y_true, y_pred):
    temp = K.log(y_pred) * y_true
    return -1 * K.sum(temp, axis=0)

# Compile
model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_loss])

#%%

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

#%%

import summary

y_pred_train = model.predict(X_train, batch_size=BATCH_SIZE)
peers_train = summary.create_summaries(docs_text_train, y_pred_train)
train_r1 = summary.evaluate_rouge_1(peers_train, sums_text_train)
train_r2 = summary.evaluate_rouge_2(peers_train, sums_text_train)

#%%

y_pred_test = model.predict(X_test, batch_size=BATCH_SIZE)
peers_test = summary.create_summaries(docs_text_test, y_pred_test)
test_r1 = summary.evaluate_rouge_1(peers_test, sums_text_test)
test_r2 = summary.evaluate_rouge_2(peers_test, sums_text_test)

