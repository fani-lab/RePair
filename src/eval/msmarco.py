import os
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import pytrec_eval
import json

def getHits(input, output,dataLocation):
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    if not searcher:
        searcher = LuceneSearcher('./../data/raw/msmarco/lucene-index.msmarco-v1-passage.20220131.9ea315/')
        if not searcher: raise ValueError("Lucene searcher cannot find/build msmarco index!")
    #get hits using ranker.
    try:
        predicted_file_list = os.listdir(input)
        for iter in predicted_file_list:
            predicted_queries = pd.read_fwf(f'{input}/{predicted_file_list[0]}', header=None)

            with open(f'{output}runs/{dataLocation}/passage_run{ iter.split(".")[0][-3:] if len(predicted_file_list) > 1 else ""}.tsv', 'w', encoding='utf-8') as retrieved_passage:
                print(f'getting relevant passages for {predicted_queries.shape[0]} queries for {iter.split(".")[0] if len(predicted_file_list) > 1 else iter}\n')
                for row in tqdm(predicted_queries.itertuples(), total=predicted_queries.shape[0]):
                    hits = searcher.search(row[1])
                    for i in range(len(hits)):
                        retrieved_passage.write(f'{i+1}\t{hits[i].docid:7}\t{hits[i].score:.5f}\n')
        compute_metric(f"./../data/raw/{dataLocation}/qrels.train.tsv", predicted_queries,retrieved_passage.name)
    except Exception as e:
        raise e


def compute_metric(qrels,predicted_queries,passage_run):
    qrels_file = pd.read_csv(qrels, sep="\t", names=['qid', 'did', 'pid', 'relevance'])
    qrels_file = qrels_file.drop(['did'], axis=1)
    qrels_dict = dict()
    passage_run_dict = dict()
    qrel_dict = {}
    with pd.read_csv(passage_run, chunksize=10, sep="\t", header=None) as reader:

            for index, row in qrels_file.iterrows():
                qrels_dict[f"{row.qid}"] = {str(row.pid): int(row.relevance)}
                run_chunk = next(reader)
                passage_run_dict[f"{row.qid}"] = dict(zip((run_chunk[1].apply(str)), run_chunk[2]))




    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'map', 'ndcg'})
    print(json.dumps(evaluator.evaluate(passage_run_dict), indent=1))
