import os
import pandas as pd
import subprocess
import time
from tqdm import tqdm

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
                    passage_run_dict['qid'].append(qrels_df["qid"][index])
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
        print('finished retrieval for all alternate queries file.\n\n\n')
    except Exception as e:
        raise e





