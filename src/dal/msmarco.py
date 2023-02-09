import json
import os
from os.path import isfile,join
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

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
    qrels.drop_duplicates(inplace=True)  # qrels have duplicates!!
    qrels.to_csv(f'{input}/qrels.train.nodups.tsv', index=False) #trec_eval does not accept duplicate rows!!
    queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
    doccol = 'docs' if cat else 'doc'
    queries_qrels[doccol] = queries_qrels['pid'].progress_apply(to_txt) #100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
    queries_qrels['ctx'] = ''
    if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False).agg({'did': list, 'pid': list, doccol: ' '.join})
    queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
    return queries_qrels

def to_norm(tf_txt):
    # lambda x: x.replace('b\'', '').replace('\'', '') if in pandas' convertors
    # TODO: we need to clean the \\x chars also
    return tf_txt.replace('b\'', '').replace('\'', '').replace('b\"', '').replace('\"', '')

def to_search(in_query, out_docids, qids, ranker='bm25', topk=100, batch=None):
    print(f'Searching docs for {in_query} ...')
    # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
    # with open(in_query, 'r', encoding='utf-8') as f: [to_docids(l) for l in f]
    queries = pd.read_csv(in_query, names=['query'], sep='\r\r', skip_blank_lines=True, engine='python')  # on windows end of line (CRLF)
    assert len(queries) == len(qids)
    # queries.dropna(inplace=True) #a change to a query might be empty str (output of t5)!!
    to_search_df(queries, out_docids, qids, ranker=ranker, topk=topk, batch=batch)

def to_search_df(queries, out_docids, qids, ranker='bm25', topk=100, batch=None):
    if ranker == 'bm25': searcher.set_bm25(0.82, 0.68)
    if ranker == 'qld': searcher.set_qld()
    with open(out_docids, 'w', encoding='utf-8') as o:
        if batch:
            for b in tqdm(range(0, len(queries), batch)):
                hits = searcher.batch_search(queries.iloc[b: b + batch]['query'].values.tolist(), qids[b: b + batch], k=topk, threads=param.settings['ncpu'])
                for qid in hits.keys():
                    for i, h in enumerate(set(hits[qid])):
                        o.write(f'{qid}\tQ0\t{h.docid:15}\t{i + 1:2}\t{h.score:.5f}\tPyserini Batch\n')
        else:
            def to_docids(row):
                if not row.query: return #in the batch call, they do the same. Also, for '', both return [], no exception
                hits = searcher.search(row.query, k=topk, remove_dups=True)
                for i, h in enumerate(hits): o.write(f'{qids[row.name]}\tQ0\t{h.docid:7}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')

            queries.progress_apply(to_docids, axis=1)

def aggregate(original, changes, output):
    metric = changes[0][1].split('.')[-1]#e.g., pred.0.1004000.bm25.map => map
    ranker = changes[0][1].split('.')[-2]#e.g., pred.0.1004000.bm25.map => bm25
    for change, metric_value in changes:
        pred = pd.read_csv(join(output, change), sep='\r\r', skip_blank_lines=False, names=[change], engine='python', index_col=False, header=None)
        assert len(original['qid']) == len(pred[change])
        pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1,2], names=['qid', f'{change}.{ranker}.{metric}'], index_col=False, skipfooter=1)
        original[change] = pred #to know the actual change
        original = original.merge(pred_metric_values, how='left', on='qid')#to know the metric value of the change
        original[change].fillna(0, inplace=True)

    print(f'Saving original queries, all their changes, and their {metric} values based on {ranker} ...')
    original.to_csv(f'{output}/{ranker}.{metric}.agg.all.tsv', sep='\t', encoding='UTF-8', index=False)

    print(f'Saving original queries, better changes, and {metric} values based on {ranker} ...')
    with open(f'{output}/{ranker}.{metric}.agg.best.tsv', mode='w', encoding='UTF-8') as agg_best:
        agg_best.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
        for index, row in tqdm(original.iterrows(), total=original.shape[0]):
            agg_best.write(f'{row.qid}\t-1\t{row.query}\t{row["original." + ranker + "." + metric]}\n')
            best_results = list()
            for change, metric_value in changes:
                if row[f'{change}.{ranker}.{metric}'] > 0 and row[f'{change}.{ranker}.{metric}'] >= row[f'original.{ranker}.{metric}']: best_results.append((row[change], row[f'{change}.{ranker}.{metric}'], change))
            best_results = sorted(best_results, key=lambda x: x[1], reverse=True)
            for i, (query, metric_value, change) in enumerate(best_results): agg_best.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')

def box(input, qrels, output):

    checks = {'gold': 'True',
              'platinum': 'golden_q_metric > original_q_metric',
              'diamond' : 'golden_q_metric > original_q_metric and golden_q_metric == 1'}
    print('Boxing datasets ...')
    ranker, metric = input.columns[-1].split('.')
    ds = {'qid': list(), 'query': list(), f'{ranker}.{metric}': list(), 'query_': list(), f'{ranker}.{metric}_': list()}

    for c in checks.keys():
        for _, group in input.groupby('qid'):
            if group.shape[0] >= 2:
                original_q, original_q_metric = group.iloc[0], group.iloc[0][f'{ranker}.{metric}']
                golden_q, golden_q_metric = group.iloc[1], group.iloc[1][f'{ranker}.{metric}']
                for i in range(1, group.shape[0]):
                    if (group.iloc[i][f'{ranker}.{metric}'] < golden_q[f'{ranker}.{metric}']): break
                    if not eval(checks[c]): break #for gold this is always true since we put >= metric values in *.agg.best.tsv
                    ds['qid'].append(original_q['qid'])
                    ds['query'].append(original_q['query'])
                    ds[f'{ranker}.{metric}'].append(original_q_metric)
                    ds['query_'].append(group.iloc[i]['query'])
                    ds[f'{ranker}.{metric}_'].append(golden_q_metric)
        df = pd.DataFrame.from_dict(ds)
        df.to_csv(f'{output}/{c}.tsv', sep='\t', encoding='utf-8', index=False, header=False)
        df.to_csv(f'{output}/{c}.initial.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'query'])
        df.to_csv(f'{output}/{c}.target.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'query_'])
        print(f'{c}  has {df.shape[0]} queries')
        qrels = df.merge(qrels, on='qid', how='inner')
        qrels.to_csv(f'{output}/{c}.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'did', 'pid', 'rel'])
        qrels.drop_duplicates(inplace=True)
        qrels.to_csv(f'{output}/{c}.qrels.nodups.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'did', 'pid', 'rel'])
