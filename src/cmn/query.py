from scipy.spatial.distance import cosine
from refinement import utils
import traceback
import os


class Query:
    """
    Query Class

    Represents a query with associated attributes and features.

    Attributes:
        qid (int): The query identifier.
        q (str): The query text.
        qrel (dict): A dictionary of tuples containing document information.
            Each tuple includes docid and relevancy, and additional information
            related to documents can be added in between.
        q_ (dict): A dictionary containing the refiner's name as the key and the tuple of (refined query, semantic similarity) as the value. Notice that type of the refined query must be Query.
        user_id (str, optional): The user identifier associated with the query.
        time (str, optional): The time of the query.
        location (str, optional): The location associated with the query.

    Args:
        qid (str): The query identifier.
        q (str): The query text.
        args (dict, optional): Additional features and attributes associated with the query,
            including 'id' for user identifier, 'time' for time information, and 'location'
            for location information.

    Example Usage:
        # Creating a Query object
        query = Query(qid='Q123', q='Sample query text', args={'id': 'U456', 'time': '2023-10-31'})

    """
    def __init__(self, domain, qid, q, qrel, parent=None, args=None):
        self.domain = domain
        self.qid = qid
        self.q = q
        self.q_ = dict()
        self.qrel = qrel
        self.qret= []
        self.parent = parent
        self.lang = 'English'

    def refine(self, model, clean=True):
        ansi_reset = "\033[0m"
        try:
            refinedq_text = model.get_refined_query(self.q)
            refinedq_text = utils.clean(refinedq_text) if clean else refinedq_text
            semsim = self.get_semsim(self.q, refinedq_text)
            print(
                f'{utils.hex_to_ansi("#F1C40F")}Info: {utils.hex_to_ansi("#3498DB")}({model.get_model_name()}){ansi_reset} {self.qid}: {self.q} -> {utils.hex_to_ansi("#52BE80")}{refinedq_text}{ansi_reset}')
        except Exception as e:
            print(
                f'{utils.hex_to_ansi("#E74C3C")}WARNING: {utils.hex_to_ansi("#3498DB")}({model.get_model_name()}){ansi_reset} Refining query [{self.qid}:{self.q}] failed!')
            print(traceback.format_exc())
            refinedq_text, semsim = self.q, 1
        refined_query = Query(domain=self.domain, qid=self.qid, q=refinedq_text, docs=self.qrel)
        self.q_[model.get_model_name()] = (refined_query, semsim)

    def search(self, ranker, searcher, topk=100):
        if not self.q: return
        # TCT_Colbert
        if ranker == 'tct_colbert':
            hits = searcher.search(self.q, k=topk)
            unique_docids = set()
            for i, h in enumerate(hits):
                if h.docid not in unique_docids: unique_docids.add(h.docid)
            if len(unique_docids) < topk: print(f'unique docids fetched less than {topk}')
        # BM25 and QLD
        else:
            hits = searcher.search(self.q, k=topk, remove_dups=True)
        self.qret.append((ranker, hits))

    def evaluate(self, in_docids, settings, output):
        #TODO use pyterrier
        metric, lib = settings['metric'], settings['treclib']
        print(f'Evaluating retrieved docs for {in_docids} with {metric} ...')
        if 'trec_eval' in lib:
            cli_cmd = f'{lib} -q -m {metric} {self.qrels} {in_docids} > {output}'
            print(cli_cmd)
            stream = os.popen(cli_cmd)
            print(stream.read())
        else:
            raise NotImplementedError

    def get_semsim(self, q1, q2):
        me, you = self.transformer_model.encode([q1, q2])
        return 1 - cosine(me, you)