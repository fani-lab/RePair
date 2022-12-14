import multiprocessing
import argparse
import time
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pandas as pd
import pytrec_eval


def addargs(parser):
    qrels = parser.add_argument_group('dataset')
    qrels.add_argument('-qrels', '--qrels-file', type=str, default=[], required=True,
                       help='qrels file location of dataset required; (eg. -qrel ./../data/raw/msmarco)')
    inference = parser.add_argument_group('infer_files')
    inference.add_argument('-infer', '--infer-file', type=str, default=[], required=True,
                           help="passage run file location;(eg. -infer ./../output/runs/msmarco/predicted_queries00.tsv")
    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/',
                        help='The output path (default: -output ./../output/metrics/msmarco)')


parser = argparse.ArgumentParser(description='calculate metrics')
addargs(parser)
args = parser.parse_args()
qrels_df = pd.read_csv(args.qrels_file, sep='\t', encoding='utf-8', names=['qid', 'did', 'pid', 'relevance'])
passage_run_df = pd.read_csv(args.infer_file, sep="\t", names=['qid', 'pid', 'score'], skip_blank_lines=False)


def create_trec_metric_from_chunk(chunks_df_source, chunks_df_target):
    passage_dict = dict()
    qrels_dict = dict()
    for qrel in chunks_df_source.itertuples():
        qrels_dict[f'{qrel.qid}'] = {str(qrel.pid): int(qrel.relevance)}
        current_qid = chunks_df_target.loc[chunks_df_target['qid'] == qrel.qid]
        passage_dict[f'{qrel.qid}'] = dict(zip(current_qid['pid'].astype(str), current_qid['score']))
        return [qrels_dict, passage_dict]


def perform_chunk_retrieval(unique_id):
    global qrels_df
    return qrels_df[qrels_df.qid == unique_id]


def perform_chunk_passage_retrieval(unique_id):
    global passage_run_df
    return passage_run_df[passage_run_df.qid == unique_id]


def compute_metric(dataLocation, passage_run):
    global passage_run_df
    global qrels_df
    print('loading files to evaluate metrics for retrieved passage using pytrec eval:\n')
    start = time.time()
    qrels_df = qrels_df.drop(['did'], axis=1)
    qrels_df = qrels_df.sort_values(by=['qid'])
    qrels_dict = dict()
    pr_dict = dict()
    print('creating chunks')
    chunks_qrels_df = process_map(perform_chunk_retrieval, qrels_df.qid.unique(),
                                  max_workers=6, chunksize=10)
    chunks_pq_df = process_map(perform_chunk_passage_retrieval, passage_run_df.qid.unique(),
                               max_workers=6, chunksize=10)
    with Pool(multiprocessing.cpu_count() - 2) as p:

        print('parallel processing the code')
        results_dfs = p.starmap(create_trec_metric_from_chunk,
                                tqdm(zip(chunks_qrels_df, chunks_pq_df), total=len(chunks_pq_df)))
        print('converting to dictionary\n')
        for result in results_dfs:
            qrels_dict.update(result[0])
            pr_dict.update(result[1])
        print('done!!\n')
        p.terminate()

    end = time.time()
    print(end - start)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'map', 'ndcg'})
    metrics_df = pd.DataFrame.from_dict(evaluator.evaluate(pr_dict)).transpose()
    metrics_df.index.name = 'qid'
    metrics_df.to_csv(f'./../output/metrics/{dataLocation}/{passage_run.split("/")[-1]}.metrics')
    print(f"finished creating metrics for {passage_run} file for {dataLocation}.")


if __name__ == '__main__':
    freeze_support()
    compute_metric(args.qrels_file.split('/')[-2], args.infer_file.split('/')[-1])
