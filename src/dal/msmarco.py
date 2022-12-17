import json

import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
tqdm.pandas()

# https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
if not searcher:
    # sometimes you need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
    searcher = LuceneSearcher(f'{input}/lucene-index.msmarco-v1-passage.20220131.9ea315/')
    if not searcher: raise ValueError(f'Lucene searcher cannot find/build msmarco index at {input}!')

def to_txt(pid):
    # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
    # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
    try: return json.loads(searcher.doc(str(pid)).raw())['contents'].lower()
    except Exception as e: raise e

def to_pair(input, output):
    queries = pd.read_csv(f'{input}/queries.train.tsv', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
    qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
    queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
    queries_qrels['doc'] = queries_qrels['pid'].progress_apply(to_txt) #100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
    queries_qrels['ctx'] = ''
    queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
    return queries_qrels


def to_search(in_query, out_docids, ranker='bm25'):
    if ranker == 'bm25': searcher.set_bm25(0.82, 0.68)
    if ranker == 'qld': searcher.set_qld()
    with open(out_docids, 'w') as o:
        def to_docids(query):
            hits = searcher.search(query, k=10, remove_dups=True)
            for i, h in enumerate(hits): o.write(f'{0} Q0  {h.docid:15} {i + 1:2}  {h.score:.5f} Pyserini \n')
        queries = pd.read_csv(in_query, skip_blank_lines=False, header=None, names=['query'], encoding='utf-8', converters={'query': str.decode('utf_8', 'strict')})
        queries['docids'] = queries['query'].progress_apply(to_docids)

to_search('../../output/t5.small.local.query.doc/pred.0-1000005', '../../output/t5.small.local.query.doc/pred.0-1000005.bm25', 'bm25')

