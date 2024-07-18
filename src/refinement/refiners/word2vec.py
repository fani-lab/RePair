import gensim
from nltk.stem import PorterStemmer

import sys
sys.path.extend(['../refinement'])

from src.refinement.refiners.abstractqrefiner import AbstractQRefiner
from src.refinement import utils


class Word2Vec(AbstractQRefiner):
    def __init__(self, vectorfile, replace=False, topn=3):
        AbstractQRefiner.__init__(self, replace, topn)
        self.vectorfile = vectorfile
        self.word2vec = None

    def get_refined_query(self, q, args=None):
        if not self.word2vec:
            print('INFO: Word2Vec: Loading word vectors in {} ...'.format(self.vectorfile))
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.vectorfile)

        upd_query = utils.get_tokenized_query(q)
        synonyms = []
        res = []
        if not self.replace:
            res = [w for w in upd_query]
        ps = PorterStemmer()
        for qw in upd_query:
            found_flag = False
            qw_stem = ps.stem(qw)
            if qw in self.word2vec.key_to_index: #in gensim 4.0 vocab change to key_to_index
                w = self.word2vec.most_similar(positive=[qw], topn=self.topn)
                for u,v in w:
                    u_stem=ps.stem(u)
                    if  u_stem!=qw_stem:
                        found_flag = True
                        res.append(u)
            if not found_flag and self.replace:
                res.append(qw)
        return super().get_refined_query(' '.join(res))


if __name__ == "__main__":
    qe = Word2Vec('../pre/wiki-news-300d-1M.vec')
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani International Crime Organization'))

    qe.replace = True
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani International Crime Organization'))
