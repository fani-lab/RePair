import traceback

import pandas as pd
from refinement import utils
from cmn.query import Query


class AbstractQRefiner:
    def __init__(self, replace=False, topn=None):
        self.replace = replace
        self.topn = topn

    # All children expanders must call this in the returning line
    def get_refined_query(self, q: str, args=None): return q

    def get_refined_query_batch(self, queries, args=None): return queries

    def get_model_name(self):
        # this is for backward compatibility for renaming this class
        if self.__class__.__name__ == 'AbstractQRefiner': return 'original'.lower()
        return f"{self.__class__.__name__.lower()}{f'.topn{self.topn}' if self.topn else ''}{'.replace' if self.replace else ''}"

    def preprocess_query(self, query, clean=True):
        ansi_reset = "\033[0m"
        try:
            q_ = self.get_refined_query(query.q, args=[query.qid])
            if 'chatgpt' in self.get_model_name(): q_ = [utils.clean(q_item) if clean else q_item for q_item in q_]
            else: q_ = utils.clean(q_) if clean else q_
            print(f'{utils.hex_to_ansi("#F1C40F")}Info: {utils.hex_to_ansi("#3498DB")}({self.get_model_name()}){ansi_reset} {query.qid}: {query.q} -> {utils.hex_to_ansi("#52BE80")}{q_}{ansi_reset}')
        except Exception as e:
            print(f'{utils.hex_to_ansi("#E74C3C")}WARNING: {utils.hex_to_ansi("#3498DB")}({self.get_model_name()}){ansi_reset} Refining query [{query.qid}:{query.q}] failed!')
            print(traceback.format_exc())
            q_ = query.q

        if isinstance(q_, list): refined_query = [Query(domain=query.domain, qid=query.qid, q=q_item, qrel=query.qrel, parent=query) for q_item in q_]
        else: refined_query = Query(domain=query.domain, qid=query.qid, q=q_, qrel=query.qrel, parent=query)
        query.q_[self.get_model_name()] = refined_query
        return query

    def preprocess_query_batch(self, queries, clean=True):
        q_s = self.get_refined_query_batch(queries)
        for q_, query in zip(q_s, queries):
            if q_:
                q_ = [utils.clean(q_) if clean else q_]
                print(f'INFO: MAIN: {self.get_model_name()}: {query.qid}: {query.q} -> {q_}')
            else:
                print(f'WARNING: MAIN: {self.get_model_name()}: Refining query [{query.qid}:{query.q}] failed!')
                print(traceback.format_exc())
                q_= query.q

            query.q_[self.get_model_name()] = q_
        return queries


    def write_queries(self, queries, outfile):
        if 'chatgpt' in self.get_model_name():
            dfs = [pd.DataFrame() for _ in range(4)]
            for query in queries:
                for i, q_item in enumerate(query.q_[self.get_model_name()]):
                    dfs[i].loc[len(dfs[i])] = [query.qid, query.q, q_item.q]
            for df in dfs: df.to_csv(f'{outfile}.pred.{i}', sep='\t', index=False)
        else:
            with open(outfile, 'w', encoding='utf-8') as file:
                for query in queries:
                    file.write(f"{query.qid}\t{query.q}\t{query.q_[self.get_model_name()].q}\n")


if __name__ == "__main__":
    qe = AbstractQRefiner()
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
