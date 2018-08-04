import nltk
import numpy as np
import random
import pickle


class DataPreprocessor:
    """

    Convert a line of text to be ready to process in main

    Attributes:
        words
    """
    def __init__(self, text):
        print "Processing text:", text
        self.words = [nltk.word_tokenize(text)]
        print "Text processed!", len(self.words), "words!"


class DataLoader:
    """

    Initialize with a string of file name
    ex: DataLoader('filename')

    Attributes:
        corpus
        words
        labels
        filename
    """
    def __init__(self, name):
        print "Opening file", name
        self.myfile = open(name, 'r')
        self.filename = name

        print "Loading data..."
        self.mydict = []
        self.corpus = []
        lines = []
        for line in self.myfile:
            self.mydict.append(line)

        for line in self.mydict:
            if line == '\r\n':
                self.corpus.append(lines)
                lines = []
            else:
                try:
                    lines.append((line.split('\t')[0], line.split('\t')[1][:-2]))  # there's [:-2] to remove newline chars
                except IndexError:
                    print "Index Error:", line

        print "Creating words and labels..."
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
        print "Data loaded!", len(self.corpus), "sentences!"

    def slice(self, percent, seed):
        random.seed(seed)
        random.shuffle(self.words)
        random.seed(seed)
        random.shuffle(self.labels)
        random.seed(seed)
        random.shuffle(self.corpus)

        num_item = int(percent * len(self.labels))
        self.corpus = self.corpus[:num_item]
        self.words = self.words[:num_item]
        self.labels = self.labels[:num_item]
        print "Data sliced!", len(self.corpus), "sentences!"

    def add(self, name):
        print "Opening file", name
        myfile = open(name, 'r')

        print "Loading data..."
        mydict = []
        corpus = []
        lines = []
        for line in myfile:
            mydict.append(line)

        for line in mydict:
            if line == '.\r\n':
                corpus.append(lines)
                self.corpus.append(lines)
                lines = []
            else:
                lines.append((line.split('\t')[0], line.split('\t')[1][:-2]))  # there's [:-2] to remove newline chars

        print "Adding words and labels..."
        for sent in corpus:
            line = []
            y_true = []
            for token in sent:
                line.append(token[0])
                y_true.append(token[1])
            self.words.append(line)
            self.labels.append(y_true)
        print "Data added!", len(self.corpus), "sentences!"

    def trim(self, length):
        print "Triming sequence length..."
        for i, line in enumerate(self.corpus):
            if len(line) > length:
                self.corpus[i] = line[:length]
        for i, line in enumerate(self.words):
            if len(line) > length:
                self.words[i] = line[:length]
        for i, line in enumerate(self.labels):
            if len(line) > length:
                self.labels[i] = line[:length]
        print "Sequence trimmed!"

class DataIndexer:
    """

    Initialize with an array of corpus to be indexed together
    ex: DataIndexer([corpus1, corpus2, ...])

    Attributes:
        index
        cnt
    """
    def __init__(self, data=[]):
        print "Indexing..."
        self.cnt = 1
        self.index = {}
        for datum in data:
            for sent in datum:
                for token in sent:
                    if token not in self.index:
                        self.index[token] = self.cnt
                        self.cnt = self.cnt + 1
        print "Data indexed!"

    def add(self, data=[]):
        print "Indexing..."
        for datum in data:
            for sent in datum:
                for token in sent:
                    if token not in self.index:
                        self.index[token] = self.cnt
                        self.cnt = self.cnt + 1
        print "Index added!"

    def save(self, name):
        with open(name + '.idx', 'wb') as fp:
            pickle.dump((self.index, self.cnt), fp)
        print "Index saved!"

    def load(self, name):
        with open(name + '.idx', 'rb') as fp:
            (self.index, self.cnt) = pickle.load(fp)
        print "Index loaded!"

class DataMapper:
    """

    Initialize with corpus and index
    ex: DataMapper(words, index)

    Attributes:
        mapped
        padded
        padsize

        oov
        oov_index
    """
    def __init__(self, data, index, verbose=True):
        self.verbose = verbose
        if verbose:
            print "Mapping..."
        self.padsize = 0
        self.mapped = []
        self.padded = []
        self.oov_index = []
        self.oov = 0
        for sent in data:
            tokens = []
            for token in sent:
                try:
                    tokens.append(index[token])
                except KeyError:
                    tokens.append(0)
                    self.oov = self.oov + 1
                    if token not in self.oov_index:
                        self.oov_index.append(token)
            self.mapped.append(tokens)
            if len(tokens) > self.padsize:
                self.padsize = len(tokens)
        if verbose:
            print "Data mapped!"

    def pad(self, size):
        if self.verbose:
            print "Padding..."
        for sent in self.mapped:
            sub = size-len(sent)
            new = np.pad(sent, (sub, 0), 'constant')
            self.padded.append(new)
        if self.verbose:
            print "Data padded!"
