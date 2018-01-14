import os
import nltk

"""
Preparing file
"""

name = raw_input('Enter file name: ')

myfile = open(name+'.conllu', 'r')

mydict = []

for line in myfile:
    mydict.append(line)

corpus = []
lines = []

for line in mydict:
    if line[:1] != '#':
        if line == '\n':
            corpus.append(lines)
            lines = []
        else:
            lines.append((nltk.word_tokenize(line)[1], nltk.word_tokenize(line)[3]))

outfile = open(name+'.pos', 'w')
for sent in corpus:
    line = []
    for token in sent[:-1]:
        w = token[0].decode('utf-8','ignore').encode("utf-8")
        outfile.write(w + '\t' + token[1] + '\r\n')
    outfile.write('-----\r\n')

outfile.close()

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