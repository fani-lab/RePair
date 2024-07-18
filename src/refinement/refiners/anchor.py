import gensim.models
from gensim.models.callbacks import CallbackAny2Vec
from src.refinement.refiners.word2vec import Word2Vec
import os
from nltk.stem import PorterStemmer
ps = PorterStemmer()
# from rdflib import Graph

# The anchor texts dataset:
# https://wiki.dbpedia.org/downloads-2016-10
# http://downloads.dbpedia.org/2016-10/core-i18n/en/anchor_text_en.ttl.bz2


class Anchor(Word2Vec):
    def __init__(self, anchorfile, vectorfile, topn=3, replace=False):
        super().__init__(vectorfile, topn=topn, replace=replace)
        self.anchorfile = anchorfile

    def train(self):

        class AnchorIter:
            def __init__(self, anchorfile):
                self.anchorfile = anchorfile

            def __iter__(self):
                for i, line in enumerate(open(self.anchorfile, encoding='utf-8')):
                    if (i % 10000 == 0 and i > 0):
                        print('INFO: ANCHOR: {} anchors have been read ...'.format(i))
                    s = line.find('> "')
                    e = line.find('"@en', s)
                    if s < 1:
                        continue
                    anchor_text = line[s + 3:e]
                    yield [ps.stem(w) for w in anchor_text.lower().split(' ')]

        class EpochLogger(CallbackAny2Vec):
            def __init__(self, epoch_count):
                self.epoch = 1
                self.epoch_count = epoch_count
            def on_epoch_begin(self, model):
                print("Epoch {}/{} ...".format(self.epoch, self.epoch_count))
                self.epoch += 1
        anchors = [anchor for anchor in AnchorIter(self.anchorfile)]#all in memory at once
        model = gensim.models.Word2Vec(anchors, size=300, sg=1, window=2, iter=100, workers=40, min_count=0, callbacks=[EpochLogger(100)])
        model.wv.save(Anchor.vectorfile)

    def get_refined_query(self, q, args=None):
        if not self.word2vec:
            if not os.path.exists(self.vectorfile):
                print('INFO: ANCHOR: Pretrained anchor vector file {} does not exist! Training has been started ...'.format(self.vectorfile))
                self.train()
            print('INFO: ANCHOR: Loading anchor vectors in {} ...'.format(self.vectorfile))
            self.word2vec = gensim.models.KeyedVectors.load(self.vectorfile, mmap='r')

        return super().get_refined_query(q)


if __name__ == "__main__":
    qe = Anchor(anchorfile='../pre/anchor_text_en.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d-100iter.vec')
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani actor International Crime Organization'))

    qe.replace = True
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani actor International Crime Organization'))
