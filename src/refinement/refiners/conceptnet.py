import requests

import sys
sys.path.extend(['../refinement'])
from nltk.stem import PorterStemmer

from refiners.abstractqrefiner import AbstractQRefiner
import utils

class Conceptnet(AbstractQRefiner):
    def __init__(self, replace=False, topn=3):
        AbstractQRefiner.__init__(self, replace, topn)

    def get_refined_query(self, q, args=None):
        upd_query = utils.get_tokenized_query(q)
        res = []
        if not self.replace:
            res = [w for w in upd_query]
        ps = PorterStemmer()
        for q in upd_query:
            q_stem = ps.stem(q)
            found_flag = False
            try:
                obj = requests.get('http://api.conceptnet.io/c/en/' + q).json()
            except:
                if self.replace:
                    res.append(q)
                continue
            if len(obj['edges']) < self.topn:
                x = len(obj['edges'])
            else:
                x = self.topn
            for i in range(x):

                try:
                    start_lan = obj['edges'][i]['start']['language']
                    end_lan = obj['edges'][i]['end']['language']
                except:
                    continue
                if obj['edges'][i]['start']['language'] != 'en' or obj['edges'][i]['end']['language'] != 'en':
                    continue
                if obj['edges'][i]['start']['label'].lower() == q:
                    if obj['edges'][i]['end']['label'] not in res and q_stem != ps.stem(obj['edges'][i]['end']['label']):
                        found_flag = True
                        res.append(obj['edges'][i]['end']['label'])
                elif obj['edges'][i]['end']['label'].lower() == q:
                    if obj['edges'][i]['start']['label'] not in res and q_stem != ps.stem(obj['edges'][i]['start']['label']):
                        found_flag = True
                        res.append(obj['edges'][i]['start']['label'])
            if not found_flag and self.replace:
                res.append(q)
        return super().get_expanded_query(' '.join(res))


if __name__ == "__main__":
    qe = Conceptnet()
    print(qe.get_model_name())
    print(qe.get_refined_query('HosseinFani International Crime Organization'))

    qe = Conceptnet(replace=True)
    print(qe.get_model_name())
    print(qe.get_refined_query('compost pile'))
