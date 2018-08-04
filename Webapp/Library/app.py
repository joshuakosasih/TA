from Main import SeqTagger

tagger = SeqTagger()

tagger.padsize =  188
tagger.char_padsize = 41
tagger.EMBEDDING_DIM = 64
tagger.CHAR_EMBEDDING_DIM = 64
tagger.patience = 2

tagger.createModel('train.txt', 'valid.txt', 'test.txt', 'polyglot.vec', 'polyglot-char.vec')

tagger.trainFit()

tagger.evaluate()

tagger.saveModel('savedweight')

tagger.loadModel('savedweight')

