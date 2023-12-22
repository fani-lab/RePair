import scipy
from nltk.stem import PorterStemmer
import numpy as np

from src.refinement.refiners.abstractqrefiner import AbstractQRefiner
from src.refinement import utils


class Glove(AbstractQRefiner):
    def __init__(self, vectorfile, replace=False, topn=3):
        AbstractQRefiner.__init__(self, replace, topn)
        self.vectorfile = vectorfile
        self.glove = None

    def get_refined_query(self, q, args=None):
        if not self.glove:
            print('INFO: Glove: Loading word vectors in {} ...'.format(self.vectorfile))
            self.glove = load_glove_model(self.vectorfile)

        upd_query = utils.get_tokenized_query(q)
        synonyms = []
        res = []
        if not self.replace:
            res = [w for w in upd_query]
        ps = PorterStemmer()
        for qw in upd_query:
            found_flag = False
            qw_stem = ps.stem(qw)
            if qw.lower() in self.glove.keys():
                w = sorted(self.glove.keys(), key=lambda word: scipy.spatial.distance.euclidean(self.glove[word], self.glove[qw]))
                w = w[:self.topn]
                for u in w:
                    u_stem = ps.stem(u)
                    if u_stem != qw_stem:
                        found_flag = True
                        res.append(u)
            if not found_flag and self.replace:
                res.append(qw)
        return super().get_refined_query(' '.join(res))


def load_glove_model(glove_file):
    with open(glove_file + ".txt", 'r', encoding='utf-8') as f:
        model = {}
        counter=0
        for line in f:
            if counter>0:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                model[word] = embedding
                counter+=1
            else:
                counter+=1
    return model


if __name__ == "__main__":
    qe = Glove('../pre/glove.6B.300d')
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani International Crime Organization'))
    qe.replace = True
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani International Crime Organization'))


