import gensim
import tagme
from src.refinement import utils
from src.refinement.refiners.word2vec import Word2Vec

tagme.GCUBE_TOKEN = "10df41c6-f741-45fc-88dd-9b24b2568a7b"

"""
@inproceedings{DBLP:conf/coling/LiZTHIS16,
  author    = {Yuezhang Li and
               Ronghuo Zheng and
               Tian Tian and
               Zhiting Hu and
               Rahul Iyer and
               Katia P. Sycara},
  editor    = {Nicoletta Calzolari and
               Yuji Matsumoto and
               Rashmi Prasad},
  title     = {Joint Embedding of Hierarchical Categories and Entities for Concept
               Categorization and Dataless Classification},
  booktitle = {{COLING} 2016, 26th International Conference on Computational Linguistics,
               Proceedings of the Conference: Technical Papers, December 11-16, 2016,
               Osaka, Japan},
  pages     = {2678--2688},
  publisher = {{ACL}},
  year      = {2016},
  url       = {https://www.aclweb.org/anthology/C16-1252/},
  timestamp = {Mon, 16 Sep 2019 17:08:53 +0200},
  biburl    = {https://dblp.org/rec/conf/coling/LiZTHIS16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


class Wiki(Word2Vec):
    def __init__(self, vectorfile, topn=3, replace=False):
        super().__init__(vectorfile=vectorfile, replace=replace, topn=topn)

    def get_concepts(self, text, score):
        concepts = tagme.annotate(text).get_annotations(score)
        res = []
        for ann in concepts:
            res.append(ann.entity_title)
        return res

    def get_refined_query(self, q, args=None):

        if not self.word2vec:
            print(f'INFO: Word2Vec: Loading word vectors in {self.vectorfile} ...')
            self.word2vec = gensim.models.KeyedVectors.load(self.vectorfile)

        query_concepts = self.get_concepts(q, 0.1)
        upd_query = utils.get_tokenized_query(q)
        res = []
        if not self.replace:
            res = [w for w in upd_query]
        for c in query_concepts:
            c_lower_e = "e_" + c.replace(" ", "_").lower()
            if c_lower_e in self.word2vec.index_to_key:
                w = self.word2vec.most_similar(positive=[c_lower_e], topn=self.topn)
                for u, v in w:
                    if u.startswith("e_"):
                        u = u.replace("e_", "")
                    elif u.startswith("c_"):
                        u = u.replace("c_", "")
                    res.append(u.replace("_", " "))

            res.append(c)
        return super().get_refined_query(' '.join(res))


if __name__ == "__main__":

    qe = Wiki(vectorfile='../pre/temp_model_Wiki')
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani actor International Crime Organization'))

    qe.replace = True
    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani actor International Crime Organization'))
