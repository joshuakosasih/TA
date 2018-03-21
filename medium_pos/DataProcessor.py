import nltk
import numpy as np


class DataLoader:
    """

    Initialize with a string of file name (without .pos)
    ex: DataLoader('filename')

    Attributes:
        corpus
        words
        labels
    """
    def __init__(self, name):
        print("Opening file", name, ".pos")
        self.myfile = open(name + '.pos', 'r')

        print("Loading data...")
        self.mydict = []
        self.corpus = []
        lines = []
        for line in self.myfile:
            self.mydict.append(line)

        for line in self.mydict:
            if line == '-----\r\n':
                self.corpus.append(lines)
                lines = []
            else:
                lines.append((line.split('\t')[0], line.split('\t')[1][:-2]))  # there's [:-2] to remove newline chars
        
        print("Creating words and labels...")
        self.words = []
        self.labels = []
        
        for sent in self.corpus:
            line = []
            y_true = []
            for token in sent:
                line.append(token[0])
                y_true.append(token[1])
            self.words.append(line)
            self.labels.append(y_true)
        print("Data loaded!", len(self.corpus), "sentences!")


class DataIndexer:
    """

    Initialize with an array of corpus to be indexed together
    ex: DataIndexer([corpus1, corpus2, ...])

    Attributes:
        index
        cnt
    """
    def __init__(self, data=[]):
        print("Indexing...")
        self.cnt = 1
        self.index = {}
        for datum in data:
            for sent in datum:
                for token in sent:
                    if token not in self.index:
                        self.index[token] = self.cnt
                        self.cnt = self.cnt + 1
        print("Data indexed!")


class DataMapper:
    """

    Initialize with corpus and index
    ex: DataMapper(words, index)

    Attributes:
        mapped
        padded
        padsize
    """
    def __init__(self, data, index, verbose=True):
        self.verbose = verbose
        if verbose:
            print("Mapping...")
        self.padsize = 0
        self.mapped = []
        self.padded = []
        self.oov = 0
        for sent in data:
            tokens = []
            for token in sent:
                try:
                    tokens.append(index[token])
                except KeyError:
                    tokens.append(0)
                    self.oov = self.oov + 1
            self.mapped.append(tokens)
            if len(tokens) > self.padsize:
                self.padsize = len(tokens)
        if verbose:
            print("Data mapped!")

    def pad(self, size):
        if self.verbose:
            print("Padding...")
        for sent in self.mapped:
            sub = size-len(sent)
            new = np.pad(sent, (sub, 0), 'constant')
            self.padded.append(new)
        if self.verbose:
            print("Data padded!")
