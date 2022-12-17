import json
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import tensorflow as tf
from pyserini.search.lucene import LuceneSearcher

import param
# https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
# searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
# sometimes you need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
searcher = LuceneSearcher(param.settings['msmarco-passage']['index'])
if not searcher: raise ValueError(f'Lucene searcher cannot find/build msmarco index at {param.settings["msmarco"]["index"]}!')

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

def to_search(in_query, out_docids, qids, ranker='bm25'):
    if ranker == 'bm25': searcher.set_bm25(0.82, 0.68)
    if ranker == 'qld': searcher.set_qld()
    with open(out_docids, 'w', encoding='utf-8') as o:
        def to_docids(row):
            query = row.query.replace('b\'', '').replace('\'', '')
            hits = searcher.search(query, k=10, remove_dups=True)
            for i, h in enumerate(hits): o.write(f'{qids[row.name]} Q0  {h.docid:15} {i + 1:2}  {h.score:.5f} Pyserini \n')
        # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
        # with open(in_query, 'r', encoding='utf-8') as f: [to_docids(l) for l in f]
        queries = pd.read_csv(in_query, names=['query'], sep='\r\n', skip_blank_lines=False, engine='python')#on windows enf of line (CRLF)
        assert len(queries) == len(qids)
        queries.progress_apply(to_docids, axis=1)

def to_search_(in_query, out_docids, qids, ranker='bm25'):
    # calling the command line and subprocess for ir
    pass



