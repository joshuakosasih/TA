import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary


embeddings_file = raw_input('Enter embedding file name: ')
wv, vocabulary = load_embeddings('polyglot.vec' if embeddings_file == '' else embeddings_file)

tsne = PCA(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(wv[:20000, :])

plt.scatter(Y[:, 0], Y[:, 1], s=1)
for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
    if random.randint(0, 39) == 0:
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show()

