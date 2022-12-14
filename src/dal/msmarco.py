import csv, json
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
tqdm.pandas()

def to_psgtxt(pid, searcher):
    # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
    # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
    try: return json.loads(searcher.doc(str(pid)).raw())['contents']
    except Exception as e: raise e

def to_pair(input, output):

    # https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    if not searcher:
        # sometimes you need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
        searcher = LuceneSearcher(f'{input}/lucene-index.msmarco-v1-passage.20220131.9ea315/')
        if not searcher: raise ValueError(f'Lucene searcher cannot find/build msmarco index at {input}!')

    queries = pd.read_csv(f'{input}/queries.train.tsv', sep='\t', index_col=False, names=['qid', 'query'], header=None)
    qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
    queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
    queries_qrels['doc'] = queries_qrels['pid'].progress_apply(to_psgtxt, args=(searcher,)) #100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
    queries_qrels['ctx'] = 'none'
    queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
    return queries_qrels
