import pickle
from DataProcessor import DataIndexer as DI

w_name = '05-17_22:39_736'

fp = open(w_name + "_1.wgt", "rb")
fout = open(w_name + "-char.vec", "w")
w = pickle.load(fp)
w = w[0]
idx = DI()
idx.load('word')
ii = idx.index
keys = ii.keys()
values = ii.values()

for i, char in enumerate(w):
    if i != 0:
        if i < idx.cnt:
            print i
            c = keys[values.index(i)]
            try:
                c.decode('utf-8')
                fout.write(c)
                for vec in char:
                    fout.write(' ' + str(vec))
                fout.write('\n')
            except UnicodeError:
                print "char is not UTF-8"
