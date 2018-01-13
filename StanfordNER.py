import os
import nltk
from nltk.tag import StanfordNERTagger
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

"""
Preparing file
"""

name = raw_input('Enter file name: ')

myfile = open(name + '.pos', 'r')

mydict = []

for line in myfile:
    mydict.append(line)

corpus = []
lines = []

for line in mydict:
    if line == '-----\r\n':
        corpus.append(lines)
        lines = []
    else:
        lines.append((nltk.word_tokenize(line)[0], nltk.word_tokenize(line)[1]))

lines = []
y_true = []

for sent in corpus:
    line = []
    for token in sent:
        line.append(token[0])
        y_true.append(token[1])
    lines.append(line)

"""
Tagging corpus
"""

entities = []
st = StanfordNERTagger('/home/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/stanford-ner/stanford-ner.jar', encoding='utf-8')
for line in lines:
    entities.append(st.tag(line))

outfile = open(name+'.lab', 'w')
for sent in entities:
    line = []
    for token in sent:
        outfile.write(token[0] + '\t' + token[1] + '\r\n')
    outfile.write('-----\r\n')

outfile.close()

y_pred = []

for line in entities:
    for token in line:
        y_pred.append(token[1])

neTypes = ['PERSON', 'LOCATION', 'ORGANIZATION']

f1_score(y_true, y_pred, average='macro')

f1_score(y_true, y_pred, labels=neTypes, average='micro')

f1_score(y_true, y_pred, average='weighted')

f1_score(y_true, y_pred, average=None)
