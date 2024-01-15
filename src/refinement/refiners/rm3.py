from pyserini.search.lucene import LuceneSearcher
from src.refinement.refiners.relevancefeedback import RelevanceFeedback


class RM3(RelevanceFeedback):
    def __init__(self, ranker, index, topn=10, topw=10, original_q_w=0.5):
        RelevanceFeedback.__init__(self, ranker=ranker, prels=None, index=index, topn=topn)
        self.topw = topw
        self.index = index
        self.ranker = ranker
        self.original_q_w = original_q_w

    def get_topn_relevant_docids(self, q=None, qid=None):
        self.searcher = LuceneSearcher(self.index)
        if self.ranker =='bm25' : self.searcher.set_bm25()
        elif self.ranker =='qld': self.searcher.set_qld()
        self.searcher.set_rm3(fb_terms=self.topw, fb_docs=self.topn, original_query_weight=self.original_q_w)
        hits = self.searcher.search(q)
        return [h.docid for h in hits]

    def get_model_name(self):
        return super().get_model_name().replace(f'topn{self.topn}', f'topn{self.topn}.{self.topw}.{self.original_q_w}')


if __name__ == "__main__":
    qe = RM3(ranker='bm25', index='./lucene-index.gov2.pos+docvectors+rawdocs')
    print(qe.get_model_name())
    print(qe.get_refined_query('Dam removal'))
