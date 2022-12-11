import os
import pandas as pd
from tqdm import tqdm
import pytrec_eval
import time
import subprocess

def getHits(input, output,dataLocation):
    qrels_file = pd.read_csv(f'./../data/raw/{dataLocation}/qrels.train.tsv', sep='\t', encoding='utf-8', names=['qid', 'did', 'pid', 'relevance'])
    qrels_file = qrels_file.drop(['did', 'relevance'], axis=1)
    #get hits using ranker.
    try:
        predicted_file_list = sorted(os.listdir(input))
        predicted_file_list = [file for file in predicted_file_list if file.endswith('.txt')]
        for pf in predicted_file_list:
            predicted_queries = pd.read_csv(f'{input}/{pf}', skip_blank_lines=False, sep="/r/r", header=None, engine='python')
            pq_run = f'{output}runs/{dataLocation}/{pf.split(".")[0] if len(predicted_file_list) > 1 else ""}.tsv'
            pq_tsv = f'{output}predictions/{dataLocation}/{pf.split(".")[0]}.tsv'
            if not os.path.isfile(pq_tsv):
                passage_run_dict = {'qid': [], 'query': []}
                start = time.time()
                print(f'getting relevant passages for {predicted_queries.shape[0]} queries for {pf.split(".")[0] if len(predicted_file_list) > 1 else pf}')
                for index, row in tqdm(predicted_queries.itertuples(), total=predicted_queries.shape[0]):
                            passage_run_dict['qid'].append(qrels_file["qid"][index])
                            passage_run_dict['query'].append(row)
                passage_run_df = pd.DataFrame().from_dict(passage_run_dict)
                passage_run_df['query'] = passage_run_df['query'].astype(str)
                passage_run_df.to_csv(pq_tsv, sep="\t", index=None, header=None)
                end = time.time()
                print(end - start)
            if not os.path.isfile(pq_run):
                print(f'retrieving passages using BM25 for alternate queries file:{pq_tsv}\n')
                start = time.time()
                subprocess.run(['python', '-m',
                                  'pyserini.search.lucene', '--index', 'msmarco-v1-passage',
                                  '--topics', pq_tsv,
                                  '--output', pq_run,
                                 '--output-format', 'msmarco',
                                  '--hits', '100', '--bm25', '--k1', '0.82', '--b', '0.68', '--batch-size', '64', '--threads', '16'])
                end = time.time()
                print(f'retrieved passages in {end - start}')
            if not os.path.isfile(f'{output}metrics/{dataLocation}/{pf.split(".")[0]}.tsv.metrics'):
                compute_metric(f'./../data/raw/{dataLocation}/qrels.train.tsv', pq_run, dataLocation)
            else:
                print(f'finished computing metrics for {pq_tsv}')
    except Exception as e:
        raise e


def compute_metric(qrels, passage_run, dataLocation):
    print('computing metrics for retrieved passages using pytrec eval:\n')
    start = time.time()
    qrels_file = pd.read_csv(qrels, sep="\t", index_col=None, header=None, names=['qid', 'did', 'pid', 'relevance'])
    passage_run_file = pd.read_csv(passage_run,sep="\t",names=['qid', 'pid', 'score'])
    # passage_run_file['score'] = passage_run_file['score'].astype(int)
    passage_run_file['pid'] = passage_run_file['pid'].astype(str)
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
    print(f"finished creating metrics for {passage_run} file for {dataLocation}.")