import os
import nltk

name = raw_input('Enter file name: ')

name = 'id-ud-test'

neTypes = ['PERSON', 'LOCATION', 'ORGANIZATION', 'O', 'Total NE']
posTypes = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

mat = []
row = []
for i in posTypes:
    for j in neTypes:
        row.append(0)
    mat.append(row)
    row = []


def updateMat(ne, pos):
    try:
        id = posTypes.index(pos)
        jd = neTypes.index(ne)
        mat[id][jd] = mat[id][jd] + 1
    except ValueError:
        print ne, pos


def printMat():
    for row in mat:
        print row


myfilePOS = open(name + '.pos', 'r')
mydictPOS = []
for line in myfilePOS:
    mydictPOS.append(line)

corpusPOS = []
lines = []
for line in mydictPOS:
    if line == '-----\r\n':
        corpusPOS.append(lines)
        lines = []
    else:
        lines.append((nltk.word_tokenize(line)[0], nltk.word_tokenize(line)[1]))

myfileNER = open(name + '.lab', 'r')
mydictNER = []
for line in myfileNER:
    mydictNER.append(line)

corpusNER = []
lines = []
for line in mydictNER:
    if line == '-----\r\n':
        corpusNER.append(lines)
        lines = []
    else:
        print line
        lines.append((nltk.word_tokenize(line)[0], nltk.word_tokenize(line)[1]))

for id, lines in enumerate(corpusPOS):
    for jd, token in enumerate(lines):
        updateMat(corpusNER[id][jd][1], corpusPOS[id][jd][1])

for idx, pos in enumerate(mat):
    pos.append(posTypes[idx])
    pos[4] = pos[0] + pos[1] + pos[2]


