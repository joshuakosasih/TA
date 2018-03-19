from inanlp.postagger.postagger import POSTagger
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

name = 'id-ud-test'
myfile = open(name + '.pos', 'r')

mydict = []
corpus = []
lines = []
for line in myfile:
    mydict.append(line)

for line in mydict:
    if line == '-----\n':
        corpus.append(lines)
        lines = []
    else:
        lines.append((line.split('\t')[0], line.split('\t')[1][:-1]))

words = []
labels = []

for sent in corpus:
    line = []
    y_true = []
    for token in sent:
        line.append(token[0])
        y_true.append(token[1])
    words.append(line)
    labels.append(y_true)

print("Data loaded!", len(corpus), "sentences!")

pt = POSTagger()

results = []
for sent in words:
    results.append(pt.tag(tokens=sent, coarse=True))

y_true = [item for sublist in labels for item in sublist]
y_pred = [item.tag for sublist in results for item in sublist]

print(classification_report(y_true, y_pred))

f1_mac = f1_score(y_true, y_pred, average='macro')
f1_mic = f1_score(y_true, y_pred, average='micro')
print('F-1 Score:')
print(max([f1_mac, f1_mic]))
