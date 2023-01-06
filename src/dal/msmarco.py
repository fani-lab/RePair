import json
from os.path import isfile,join
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import tensorflow as tf
from pyserini.search.lucene import LuceneSearcher

import param
# https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
# searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
# sometimes you need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
searcher = LuceneSearcher(param.settings['msmarco.passage']['index'])
if not searcher: raise ValueError(f'Lucene searcher cannot find/build msmarco.passage index at {param.settings["msmarco.passage"]["index"]}!')

def to_txt(pid):
    # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
    # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
    try: return json.loads(searcher.doc(str(pid)).raw())['contents'].lower()
    except Exception as e: raise e

def to_pair(input, output, cat=True):
    queries = pd.read_csv(f'{input}/queries.train.tsv', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
    qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
    queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
    doccol = 'docs' if cat else 'doc'
    queries_qrels[doccol] = queries_qrels['pid'].progress_apply(to_txt) #100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
    queries_qrels['ctx'] = ''
    if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False).agg({'did': list, 'pid': list, doccol: ' '.join})
    queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
    #removes duplicates from qrels file
    if not isfile(join(input, 'qrels.train.nodups.tsv')):
        qrels.drop_duplicates(inplace=True)
        qrels.to_csv(f'{input}/qrels.train.nodups.tsv', sep='\t', index=False, header=None)
    return queries_qrels

def to_norm(tf_txt):
    # lambda x: x.replace('b\'', '').replace('\'', '') if in pandas' convertors
    # TODO: we need to clean the \\x chars also
    return tf_txt.replace('b\'', '').replace('\'', '').replace('b\"', '').replace('\"', '')

def to_search(in_query, out_docids, qids, ranker='bm25', topk=100, batch=None):
    print(f'Searching docs for {in_query} ...')
    # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
    # with open(in_query, 'r', encoding='utf-8') as f: [to_docids(l) for l in f]
    queries = pd.read_csv(in_query, names=['query'], converters={'query': to_norm}, sep='\r\r', skip_blank_lines=False, engine='python')  # on windows enf of line (CRLF)
    to_search_df(queries, out_docids, qids, ranker=ranker, topk=topk, batch=batch)

def to_search_df(queries, out_docids, qids, ranker='bm25', topk=100, batch=None):
    if ranker == 'bm25': searcher.set_bm25(0.82, 0.68)
    if ranker == 'qld': searcher.set_qld()
    assert len(queries) == len(qids)
    if batch:
        with open(out_docids, 'w', encoding='utf-8') as o:
            for b in tqdm(range(0, len(queries), batch)):
                hits = searcher.batch_search(queries.iloc[b: b + batch]['query'].values.tolist(), qids[b: b + batch], k=topk, threads=4)
                for qid in hits.keys():
                    for i, h in enumerate(hits[qid]):
                        o.write(f'{qid}\tQ0\t{h.docid:15}\t{i + 1:2}\t{h.score:.5f}\tPyserini Batch\n')
    else:
        with open(out_docids, 'w', encoding='utf-8') as o:
            def to_docids(row):
                 if not pd.isna(row.query):
                    hits = searcher.search(row.query, k=topk, remove_dups=True)
                    for i, h in enumerate(hits): o.write(f'{qids[row.name]}\tQ0\t{h.docid:7}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')

            queries.progress_apply(to_docids, axis=1)

def aggregate(original,prediction_files_list,output):
    for file, file_map in prediction_files_list:
        pred_df = pd.read_csv(join(output, file), sep='\r\r', skip_blank_lines=False,
                              names=[f'{file}_query'], engine='python', index_col=False)
        assert len(original['qid']) == len(pred_df[f'{file}_query'])
        pred_df['qid'] = original['qid']
        pred_df_map = pd.read_csv(join(output, file_map), sep='\t', names=['map', 'qid', f'{file}_map'],
                                  index_col=False, low_memory=False)
        pred_df_map.drop(columns=['map'], inplace=True)
        pred_df_map.drop(pred_df_map.tail(1).index, inplace=True)
        original = original.merge(pred_df, how='left', on='qid')
        original = original.merge(pred_df_map, how='left', on='qid')

    print('saving all merged queries\n')
    original.to_csv(f'{output}/agg.all.tsv', sep='\t', encoding='utf-8', index=False)
    print('calculating performance of predicted queries\n')
    with open(f'{output}/agg.best.tsv', mode='w', encoding='UTF-8') as agg_best:
        agg_best.write('qid\torder\tquery\tmap\n')
        for index, row in original.iterrows():
            agg_best.write(f'{row.qid}\t-1\t{row.query}\t{row.og_map}\n')
            best_results = list()
            for i in range(1, 25):
                if row[f'pred.{i}-1004000_map'] >= row['og_map']:
                    best_results.append((row[f'pred.{i}-1004000_query'], row[f'pred.{i}-1004000_map'],f'pred.{i}'))
            best_results = sorted(best_results, key=lambda x: x[1], reverse=True)
            for i, (a, b) in enumerate(best_results): agg_best.write(f'{row.qid}\t{i+1}\t{a}\t{b}\n')
    print('saving file for all predicted queries that performed better than the original query\n')
    return 0
