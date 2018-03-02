"""
Preparing file
"""

name = raw_input('Enter file name: ')

myfile = open(name+'.conllu', 'r')

mydict = []

print "Processing..."

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
            lines.append((line.split('\t')[1], line.split('\t')[3]))

outfile = open(name+'.pos', 'w')
wi = 1
label_index = {}
for sent in corpus:
    line = []
    for token in sent[:-1]:
        w = token[0].decode('utf-8','ignore').encode("utf-8")
        outfile.write(w + '\t' + token[1] + '\r\n')
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
