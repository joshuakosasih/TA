import anago
from anago.reader import load_data_and_labels

x_train, y_train = load_data_and_labels('train.txt')
x_valid, y_valid = load_data_and_labels('valid.txt')
x_test, y_test = load_data_and_labels('test.txt')

# model = anago.Sequence()
model = anago.Sequence(char_emb_size=100,word_emb_size=50,char_lstm_units=25,word_lstm_units=100,dropout=0.5,char_feature=True,crf=True,batch_size=3,optimizer='adam', learning_rate=0.005,lr_decay=0.7,clip_gradients=5.0)
model.train(x_train, y_train, x_valid, y_valid)

model.eval(x_test, y_test)

matres = []
for sent in x_test:
	res = model.analyze(sent)['entities']
	matres.append(res)

y_resu = []
for i, sent in enumerate(matres):
	sent_pred = ['O']*len(y_test[i])
	for enti in sent:
		bo = enti['beginOffset']
		sent_pred[bo] = 'B-'+enti['type']
		for x in range(bo+1, enti['endOffset']):
			sent_pred[x] = 'I-'+enti['type']
	y_resu.append(sent_pred)

y_true = [item for sublist in y_test for item in sublist]
y_pred = [item for sublist in y_resu for item in sublist]

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

myset = set(y_true)
mynewlist = list(myset)
mynewlist.sort()
labels = mynewlist[:-1]

from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average='macro', labels=labels)
f1_score(y_true, y_pred, average='micro', labels=labels)
