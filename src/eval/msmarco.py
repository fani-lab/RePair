import os
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import pytrec_eval
import time
import numpy as np
import feather
def getHits(input, output,dataLocation):
    print('retrieving passages using BM25 for alternate queries:\n')
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    qrels_file = pd.read_csv(f'./../data/raw/{dataLocation}/qrels.train.tsv', sep='\t', encoding='utf-8', names=['qid', 'did', 'pid', 'relevance'])
    qrels_file = qrels_file.drop(['did', 'relevance'], axis=1)
    passage_run_dict = {'qid': [], 'pid': [], 'score': []}
    if not searcher:
        searcher = LuceneSearcher('./../data/raw/msmarco/lucene-index.msmarco-v1-passage.20220131.9ea315/')
        if not searcher: raise ValueError("Lucene searcher cannot find/build msmarco index!")
    #get hits using ranker.
    try:
        predicted_file_list = os.listdir(input)
        for pf in predicted_file_list:
            predicted_queries = pd.read_csv(f'{input}/{pf}', skip_blank_lines=False, sep="/r/r", header=None, engine='python')
            if not os.path.isfile(f'{output}runs/{dataLocation}/passage_run{pf.split(".")[0][-2:]}.feather'):
                start = time.time()
                print(f'getting relevant passages for {predicted_queries.shape[0]} queries for {pf.split(".")[0] if len(predicted_file_list) > 1 else pf}\n')
                for index,row in tqdm(predicted_queries.itertuples(), total=predicted_queries.shape[0]):
                    if not isinstance(row, float):
                        hits = searcher.search(row)
                        for i in range(len(hits)):
                            passage_run_dict['qid'].append(qrels_file["qid"][index])
                            passage_run_dict['pid'].append(f'{hits[i].docid:7}')
                            passage_run_dict['score'].append(f'{hits[i].score:.5f}')
                    else:
                        for i in range(10):
                            passage_run_dict['qid'].append(qrels_file["qid"][index])
                            passage_run_dict['pid'].append(str(0))
                            passage_run_dict['score'].append(str(0))
                passage_run_df = pd.DataFrame().from_dict(passage_run_dict)
                passage_run_df.to_feather(f'{output}runs/{dataLocation}/passage_run{pf.split(".")[0][-2:] if len(predicted_file_list) > 1 else ""}.feather')
                # with open(f'{output}runs/{dataLocation}/passage_run{pf.split(".")[0][-2:] if len(predicted_file_list) > 1 else ""}.tsv', 'w', encoding='utf-8') as retrieved_passage:
                #     print(f'getting relevant passages for {predicted_queries.shape[0]} queries for {pf.split(".")[0] if len(predicted_file_list) > 1 else pf}\n')
                #     for index, row in tqdm(predicted_queries.itertuples(), total=predicted_queries.shape[0]):
                #         if not isinstance(row, float):
                #             hits = searcher.search(row)
                #             for i in range(len(hits)):
                #                 retrieved_passage.write(f'{qrels_file["qid"][index]}\t{hits[i].docid:7}\t{hits[i].score:.5f}\n')
                #         else:
                #             for i in range(10):
                #                 retrieved_passage.write(f'\t{qrels_file["qid"][index]}\t{int(0)}\t{0}\n')
                end = time.time()
                print(end - start)
            if not os.path.isfile(f'./../output/metrics/{dataLocation}/passage_run{pf.split(".")[0][-2:] if len(predicted_file_list) > 1 else ""}.feather.metrics'):
                compute_metric(f'./../data/raw/{dataLocation}/qrels.train.tsv', f'{output}runs/{dataLocation}/passage_run{pf.split(".")[0][-2:] if len(predicted_file_list) > 1 else ""}.feather', dataLocation)
            else:
                print(f'finished computing metrics for ./../output/metrics/{dataLocation}/passage_run{pf.split(".")[0][-2:] if len(predicted_file_list) > 1 else ""}.feather.metrics')
    except Exception as e:
        raise e


def compute_metric(qrels, passage_run, dataLocation):
    print('computing metrics for retrieved passages using pytrec eval:\n')
    start = time.time()
    qrels_file = pd.read_csv(qrels, sep="\t", index_col=None, header=None, names=['qid', 'did', 'pid', 'relevance'])
    passage_run_file = pd.read_feather(passage_run)
    passage_run_file['score'] = passage_run_file['score'].astype(np.float32)
    print(passage_run_file.dtypes)
    qrels_file = qrels_file.drop(['did'], axis=1)
    qrels_dict = dict()
    passage_run_dict = dict()
    for row in tqdm(qrels_file.itertuples(), total=qrels_file.shape[0]):
        qrels_dict[f"{row.qid}"] = {str(row.pid): int(row.relevance)}
        current_qid = passage_run_file.loc[passage_run_file['qid'] == row.qid]
        passage_run_dict[f"{row.qid}"] = dict(zip((current_qid['pid']), current_qid['score']))
    end = time.time()
    print(end - start)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'map', 'ndcg'})
    metrics_df = pd.DataFrame.from_dict(evaluator.evaluate(passage_run_dict)).transpose()
    metrics_df.index.name = 'qid'
    metrics_df.to_csv(f'./../output/metrics/{dataLocation}/{passage_run.split("/")[-1]}.metrics')
    print(f"finished creating metrics for {passage_run} file for {dataLocation}.\n")

