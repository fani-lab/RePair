import traceback
from src.dal.query import Query
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import sys
sys.path.extend(['../src'])
from src.refinement import utils
from src.refinement.refiner_param import settings


class AbstractQRefiner:
    def __init__(self, replace=False, topn=None):
        self.transformer_model = SentenceTransformer(settings['transformer_model'])
        self.replace = replace
        self.topn = topn

    # All children expanders must call this in the returning line
    def get_refined_query(self, q: Query, args=None): return q.q

    def get_refined_query_batch(self, queries, args=None): return queries, [1] * len(queries)

    def get_model_name(self):
        # this is for backward compatibility for renaming this class
        if self.__class__.__name__ == 'AbstractQRefiner': return 'original.queries'.lower()
        return f"{self.__class__.__name__.lower()}{f'.topn{self.topn}' if self.topn else ''}{'.replace' if self.replace else ''}"

    def preprocess_query(self, query, clean=True):
        ansi_reset = "\033[0m"
        try:
            q_ = self.get_refined_query(query)
            q_ = utils.clean(q_) if clean else q_
            semsim = self.get_semsim(query, q_)
            print(f'{utils.hex_to_ansi("#F1C40F")}Info: {utils.hex_to_ansi("#3498DB")}({self.get_model_name()}){ansi_reset} {query.qid}: {query.q} -> {utils.hex_to_ansi("#52BE80")}{q_}{ansi_reset}')
        except Exception as e:
            print(f'{utils.hex_to_ansi("#E74C3C")}WARNING: {utils.hex_to_ansi("#3498DB")}({self.get_model_name()}){ansi_reset} Refining query [{query.qid}:{query.q}] failed!')
            print(traceback.format_exc())
            q_, semsim = query.q, 1

        query.q_[self.get_model_name()] = (q_, semsim)
        return query

    def preprocess_query_batch(self, queries, clean=True):
        q_s, semsims = self.get_refined_query_batch(queries)
        for q_, semsim, query in zip(q_s, semsims, queries):
            if q_:
                q_ = [utils.clean(q_) if clean else q_]
                semsim = self.get_semsim(query, q_)
                print(f'INFO: MAIN: {self.get_model_name()}: {query.qid}: {query.q} -> {q_}')
            else:
                print(f'WARNING: MAIN: {self.get_model_name()}: Refining query [{query.qid}:{query.q}] failed!')
                print(traceback.format_exc())
                q_, semsim = query.q, 1

            query.q_[self.get_model_name()] = (q_, semsim)
        return queries

    '''
    Calculates the difference between the original and back-translated query
    '''
    def get_semsim(self, q1, q2):
        me, you = self.transformer_model.encode([q1, q2])
        return 1 - cosine(me, you)

    def write_queries(self, queries, outfile):
        with open(outfile, 'w', encoding='utf-8') as file:
            # file.write(f"qid\tq\tq_\tsemsim\n")
            for query in queries:
                file.write(f"{query.qid}\t{query.q}\t{query.q_[self.get_model_name()][0]}\t{query.q_[self.get_model_name()][1]}\n")


if __name__ == "__main__":
    qe = AbstractQRefiner()
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
