import multiprocessing
import os
import pandas as pd
import pytrec_eval
import subprocess
import time
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

qrels_df = pd.DataFrame()
passage_run_df = pd.DataFrame()

def create_trec_metric_from_chunk(chunks_df_source, chunks_df_target):
    passage_dict = dict()
    qrels_dict = dict()
    for qrel in chunks_df_source.itertuples():
        qrels_dict[f'{qrel.qid}'] = {str(qrel.pid): int(chunks_df_source.relevance)}
        current_qid = chunks_df_target.loc[chunks_df_target['qid'] == qrel.qid]
        passage_dict[f'{qrel.qid}'] = dict(zip(current_qid['pid'], current_qid['score']))
        return [qrels_dict, passage_dict]


def getHits(input, output, dataLocation):
    qrels_df = pd.read_csv(f'./../data/raw/{dataLocation}/qrels.train.tsv', sep='\t', encoding='utf-8',
                           names=['qid', 'did', 'pid', 'relevance'])
    qrels_df = qrels_df.drop(['did'], axis=1)
    # get hits using ranker.
    try:
        predicted_file_list = sorted(os.listdir(input))
        predicted_file_list = [file for file in predicted_file_list if file.endswith('.txt')]
        for pf in predicted_file_list:
            predicted_queries = pd.read_csv(f'{input}/{pf}', skip_blank_lines=False, sep="/r/r", header=None,
                                            engine='python')
            pq_run = f'{output}runs/{dataLocation}/{pf.split(".")[0] if len(predicted_file_list) > 1 else ""}.tsv'
            pq_tsv = f'{output}predictions/{dataLocation}/{pf.split(".")[0]}.tsv'
            if not os.path.isfile(pq_tsv):
                passage_run_dict = {'qid': [], 'query': []}
                start = time.time()
                print(
                    f'getting relevant passages for {predicted_queries.shape[0]} queries for {pf.split(".")[0] if len(predicted_file_list) > 1 else pf}')
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
                                '--hits', '100', '--bm25', '--k1', '0.82', '--b', '0.68', '--batch-size', '64',
                                '--threads', '16'])
                end = time.time()
                print(f'retrieved passages in {end - start}')
            if not os.path.isfile(f'{output}metrics/{dataLocation}/{pf.split(".")[0]}.tsv.metrics'):
                compute_metric(f'./../data/raw/{dataLocation}/qrels.train.tsv', pq_run, dataLocation)
            else:
                print(f'finished computing metrics for {pq_tsv}')
    except Exception as e:
        raise e


def perform_chunk_retrieval(unique_id):
    global qrels_df
    return qrels_df[qrels_df.qid == unique_id]


def perform_chunk_passage_retrieval(unique_id):
    global passage_run_df
    return passage_run_df[passage_run_df.qid == unique_id]

def init_worker(q_f, pr_df):
    print('initialize worker for global passage')
    global qrels_df
    global passage_run_df
    qrels_df = q_f
    passage_run_df = pr_df



def compute_metric(qrels, passage_run, dataLocation):
    print('loading files to evaluate metrics for retrieved passage using pytrec eval:\n')
    start = time.time()
    qrels_df = pd.read_csv(qrels, sep='\t', encoding='utf-8',
                           names=['qid', 'did', 'pid', 'relevance'])
    qrels_df = qrels_df.drop(['did'], axis=1)
    passage_run_df = pd.read_csv(passage_run, sep="\t", names=['qid', 'pid', 'score'])
    qrels_df = qrels_df.sort_values(by=['qid'])
    passage_run_df['pid'] = passage_run_df['pid'].astype(str)
    qrels_dict = dict()
    pr_dict = dict()
    pool = Pool(multiprocessing.cpu_count() - 1, initializer=init_worker, initargs=(qrels_df, passage_run_df,))
    with pool as p:
        print('creating chunks')
        chunks_qrels_df = process_map(perform_chunk_retrieval, qrels_df.qid.unique(), max_workers=6, chunksize=10)
        chunks_pq_df = process_map(perform_chunk_passage_retrieval, pq_df.qid.unique(), max_workers=6, chunksize=100)
        print('parallel processing the code')
        results_dfs = tqdm(p.starmap(create_trec_metric_from_chunk, zip(chunks_qrels_df, chunks_pq_df)),
                           total=len(chunks_qrels_df))
        print('converting to dictionary\n')
        for result in results_dfs:
            qrels_dict.update(result[0])
            pr_dict.update(result[1])
        print('done!!\n')
        p.close()
        p.join()

    end = time.time()
    print(end - start)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'map', 'ndcg'})
    metrics_df = pd.DataFrame.from_dict(evaluator.evaluate(pr_dict)).transpose()
    metrics_df.index.name = 'qid'
    metrics_df.to_csv(f'./../output/metrics/{dataLocation}/{passage_run.split("/")[-1]}.metrics')
    print(f"finished creating metrics for {passage_run} file for {dataLocation}.")
