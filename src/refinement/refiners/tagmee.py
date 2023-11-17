import tagme
tagme.GCUBE_TOKEN = "10df41c6-f741-45fc-88dd-9b24b2568a7b"

import sys, os
sys.path.extend(['../refinement'])

from refiners.abstractqrefiner import AbstractQRefiner
import utils

class Tagmee(AbstractQRefiner):
    def __init__(self, topn=3, replace=False):
        AbstractQRefiner.__init__(self, replace, topn)

    def get_concepts(self, text, score):
        concepts = tagme.annotate(text).get_annotations(score)
        res = []
        for ann in concepts:
            res.append(ann.entity_title)
        return res

    def get_refined_query(self, q, args=None):

        query_concepts = self.get_concepts(q, 0.1)
        upd_query = utils.get_tokenized_query(q)
        res = []
        if not self.replace:
            res = [w for w in upd_query]
        for c in query_concepts:
            res.append(c)
        return super().get_refined_query(' '.join(res))


if __name__ == "__main__":
    qe = Tagmee()

    print(qe.get_model_name())
    print(qe.get_refined_query('HosseinFani actor International Crime Organization'))


    qe = Tagmee(replace=True)
    print(qe.get_model_name())
    print(qe.get_refined_query('Magnetic Levitation-Maglev'))

