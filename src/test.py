import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, Process
import multiprocessing
import time, pytrec_eval
cpu = multiprocessing.cpu_count() - 1

def euclidian_distance(chunks_df_source,chunks_df_target):
    passage_dict = dict()
    qrels_dict = dict()
    for qrel in chunks_df_source.itertuples():
        qrels_dict[f'{qrel.qid}'] = {str(qrel.pid): int(chunks_df_source.relevance)}
        current_qid = chunks_df_target.loc[chunks_df_target['qid'] == qrel.qid]
        passage_dict[f'{qrel.qid}'] = dict(zip(current_qid['pid'].astype(str), current_qid['score']))
        return [qrels_dict, passage_dict]

# def perform_chunk_retrieval(unique_id,df):
#     global chunk_qrels_df
#     chunk_qrels_df.append(df[df.qid == unique_id])

def main():
    start = time.time()
    print('loading files')
    qrels_file = './../Data/raw/msmarco/qrels.train.tsv'
    run_file = './../output/runs/msmarco/predicted_queries00.tsv'
    qrels_df = pd.read_csv(qrels_file, sep='\t', encoding='utf-8', names=['qid', 'did', 'pid', 'relevance'])
    qrels_df = qrels_df.sort_values(by=['qid'],)
    pq_df = pd.read_csv(run_file, sep="\t", names=['qid', 'pid', 'score'])
    qrels_dict = dict()
    pr_dict = dict()
    print('creating chunks')

    chunks_qrels_df = [qrels_df[qrels_df.qid == qid] for qid in tqdm(qrels_df.qid.unique())]
    chunks_pq_df = [pq_df[pq_df.qid == qid] for qid in tqdm(pq_df.qid.unique())]
    print('parallel processing the code')
    with Pool(cpu) as p:
        results_dfs = tqdm(p.starmap(euclidian_distance, zip(chunks_qrels_df, chunks_pq_df)), total=len(chunks_qrels_df))
        print('converting to dictionary\n')
        for result in results_dfs:
            qrels_dict.update(result[0])
            pr_dict.update(result[1])
        print('done!!\n')
        p.close()
        p.join()
    print(qrels_dict)
    print(pr_dict)
    end = time.time()
    print(f'completed everything {end - start}')

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'map', 'ndcg'})
    metrics_df = pd.DataFrame.from_dict(evaluator.evaluate(pr_dict)).transpose()
    metrics_df.index.name = 'qid'
    metrics_df.to_csv(f'./../output/metrics/msmarco/multiprocess.test.metrics')
    print(f"finished creating metrics for {run_file} file.")

if __name__ == '__main__':

    freeze_support()
    main()
