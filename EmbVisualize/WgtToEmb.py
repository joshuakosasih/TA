import pickle
from DataProcessor import DataIndexer as DI

w_name = '05-17_13:37_921'

fp = open(w_name + "_5.wgt", "rb")
fout = open(w_name + ".vec", "w")
w = pickle.load(fp)
w = w[0]
word = DI()
word.load('word')
ci = word.index
keys = ci.keys()
values = ci.values()

for i, char in enumerate(w):
    if i != 0:
        c = keys[values.index(i)]
        try:
            c.decode('utf-8')
            fout.write(c)
            for vec in char:
                fout.write(' ' + str(vec))
            fout.write('\n')
        except UnicodeError:
            print "char is not UTF-8"
