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

myfile = open(name+'.txt', 'r')

print "Processing..."

mydict = []

for line in myfile:
    mydict.append(line)

lines = []

for line in mydict:
    lines.append(nltk.word_tokenize(line))

corpus = []
neTypes = ['PERSON', 'LOCATION', 'ORGANIZATION']

for line in lines:
    neStat = 0 # 0-not NE tag, 1-begin NE tag, 2-end NE tag
    label = 'O'
    structured = []
    for token in line:
        if neStat == 1:
            if token == '>':
                neStat = 0
            elif token in neTypes:
                label = token
        elif neStat == 2:
            if token == '>':
                neStat = 0
        else:
            if token == '<':
                if label == 'O':
                    neStat = 1
                else:
                    label = 'O'
                    neStat = 2
            else:
                structured.append((token, label))
    corpus.append(structured)

outfile = open(name+'.ner', 'w')
wi = 1
label_index = {}
for sent in corpus:
    line = []
    for token in sent[:-1]:
        outfile.write(token[0] + '\t' + token[1] + '\r\n')
        if token[1] not in label_index:
            label_index[token[1]] = wi
            wi = wi + 1
    outfile.write('-----\r\n')

outfile.close()

print "Done!"
print "Label Index:"
print label_index

"""
lines = []
y_true = []

for sent in corpus:
    line = []
    for token in sent:
        line.append(token[0])
        y_true.append(token[1])
    lines.append(line)
"""
