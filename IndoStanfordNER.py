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

myfile = open(name + '.ner', 'r')

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
        print line
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
st = StanfordNERTagger('/home/stanford-ner/ner-indo.ser.gz', '/home/stanford-ner/stanford-ner.jar', encoding='utf-8')
print 'Tagging...'
for line in lines:
    entities.append(st.tag(line))

print 'Done'

outfile = open(name+'.lab', 'w')
for sent in entities:
    line = []
    for token in sent:
        if token[0] == '':
            w = '/'
        else:
            w = ''
        for x in token[0]:
            if ord(x) < 128:
                w = w + x
            else:
                w = w + '~'
        #w = ''.join([x for x in token[0] if ord(x) < 128])
        outfile.write(w + '\t' + token[1] + '\r\n')
    outfile.write('-----\r\n')

outfile.close()

y_pred = []

for line in entities:
    for token in line:
        y_pred.append(token[1])

neTypes = ['O', 'PERSON', 'LOCATION', 'ORGANIZATION']

print "Macro", f1_score(y_true, y_pred, labels=neTypes, average='macro')

print "Micro", f1_score(y_true, y_pred, labels=neTypes, average='micro')

print f1_score(y_true, y_pred, average='weighted')

print f1_score(y_true, y_pred, average=None)

from sklearn.metrics import classification_report

print "Sklearn evaluation:"
print classification_report(y_true, y_pred, labels=neTypes)

print "Without O"

neTypes = ['PERSON', 'LOCATION', 'ORGANIZATION']

print "Sklearn evaluation:"
print classification_report(y_true, y_pred, labels=neTypes)
