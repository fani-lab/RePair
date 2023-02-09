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
    ir_method = changes[0][1].split('.')[-2]#e.g., pred.0.1004000.bm25.map => bm25
    for change, metric_value in changes:
        pred = pd.read_csv(join(output, change), sep='\r\r', skip_blank_lines=False, names=[change], engine='python', index_col=False, header=None)
        assert len(original['qid']) == len(pred[change])
        pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1,2], names=['qid', f'{change}.value'], index_col=False, skipfooter=1)
        original[change] = pred #to know the actual change
        original = original.merge(pred_metric_values, how='left', on='qid')#to know the metric value of the change
        original[change].fillna(0, inplace=True)

    print(f'Saving original queries, all their changes, and their {metric} values based on {ir_method} ...')
    original.to_csv(f'{output}/{ir_method}.{metric}.agg.all.tsv', sep='\t', encoding='UTF-8', index=False)

    print(f'Saving original queries, better changes, and {metric} values based on {ir_method} ...')
    with open(f'{output}/{ir_method}.{metric}.agg.best.tsv', mode='w', encoding='UTF-8') as agg_best:
        agg_best.write(f'qid\torder\tquery\t{metric}\n')
        for index, row in tqdm(original.iterrows(), total=original.shape[0]):
            agg_best.write(f'{row.qid}\t-1\t{row.query}\t{row["original.value"]}\n')
            best_results = list()
            for change, metric_value in changes:
                if row[f'{change}.value'] >= row['original.value']: best_results.append((row[change], row[f'{change}.value'], change))
            best_results = sorted(best_results, key=lambda x: x[1], reverse=True)
            for i, (query, metric_value, change) in enumerate(best_results): agg_best.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')

def box(input, qrels, output):
    ds = {'diamond': {'qid': list(), 'i_query': list(), 'i_map': list(), 'f_query': list(), 'f_map': list()},
          'platinum': {'qid': list(), 'i_query': list(), 'i_map': list(), 'f_query': list(), 'f_map': list()},
          'gold': {'qid': list(), 'i_query': list(), 'i_map': list(), 'f_query': list(), 'f_map': list()}}

    print('Creating diamond, platinum and gold datasets ...')
    for _, group in input.groupby('qid'):
        df = group.head(2)
        if df.shape[0] == 2:
            # diamond_queries
            if (df.iloc[1]['map'] == 1 and df.iloc[0]['map'] < 1):
                ds['diamond']['qid'].append(df.iloc[0]['qid'])
                if (len(df.iloc[0][2]) < 7):
                    ds['diamond']['i_query'].append(group.tail(1)['query'].values[0])
                    ds['diamond']['i_map'].append(group.tail(1)['map'].values[0])
                else:
                    ds['diamond']['i_query'].append(df.iloc[0]['query'])
                    ds['diamond']['i_map'].append(df.iloc[0]['map'])

                ds['diamond']['f_query'].append(df.iloc[1]['query'])
                ds['diamond']['f_map'].append(df.iloc[1]['map'])
            # platinum queries
            if (df.iloc[1]['map'] > df.iloc[0]['map']):
                ds['platinum']['qid'].append(df.iloc[0]['qid'])
                if (len(df.iloc[0][2]) < 7):
                    ds['platinum']['i_query'].append(group.tail(1)['query'].values[0])
                    ds['platinum']['i_map'].append(group.tail(1)['map'].values[0])
                else:
                    ds['platinum']['i_query'].append(df.iloc[0]['query'])
                    ds['platinum']['i_map'].append(df.iloc[0]['map'])
                ds['platinum']['f_query'].append(df.iloc[1]['query'])
                ds['platinum']['f_map'].append(df.iloc[1]['map'])
            # gold_queries
            if (df.iloc[1]['map'] >= df.iloc[0]['map']):
                ds['gold']['qid'].append(df.iloc[0]['qid'])
                if (len(df.iloc[0][2]) < 7):
                    ds['gold']['i_query'].append(group.tail(1)['query'].values[0])
                    ds['gold']['i_map'].append(group.tail(1)['map'].values[0])
                else:
                    ds['gold']['i_query'].append(df.iloc[0]['query'])
                    ds['gold']['i_map'].append(df.iloc[0]['map'])
                ds['gold']['f_query'].append(df.iloc[1]['query'])
                ds['gold']['f_map'].append(df.iloc[1]['map'])
        else:
            if(df.iloc[0]['map'] == 1):
                ds['diamond']['qid'].append(df.iloc[0]['qid'])
                ds['diamond']['i_query'].append(df.iloc[0]['query'])
                ds['diamond']['i_map'].append(df.iloc[0]['map'])
                ds['diamond']['f_query'].append(df.iloc[0]['query'])
                ds['diamond']['f_map'].append(df.iloc[0]['map'])
            if (df.iloc[0]['map'] > 0):
                ds['platinum']['qid'].append(df.iloc[0]['qid'])
                ds['platinum']['i_query'].append(df.iloc[0]['query'])
                ds['platinum']['i_map'].append(df.iloc[0]['map'])
                ds['platinum']['f_query'].append(df.iloc[0]['query'])
                ds['platinum']['f_map'].append(df.iloc[0]['map'])
            ds['gold']['qid'].append(df.iloc[0]['qid'])
            ds['gold']['i_query'].append(df.iloc[0]['query'])
            ds['gold']['i_map'].append(df.iloc[0]['map'])
            ds['gold']['f_query'].append(df.iloc[0]['query'])
            ds['gold']['f_map'].append(df.iloc[0]['map'])

    for d in ds.keys():
        df = pd.DataFrame.from_dict(ds[d])
        df.to_csv(f'{output}/datasets/{d}.tsv', sep='\t', encoding='utf-8', index=False, header=False)
        df.to_csv(f'{output}/datasets/{d}_initial.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'i_query'])
        df.to_csv(f'{output}/datasets/{d}_target.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'f_query'])
        print(f'{d}  has {df.shape[0]} queries')
        qrels = df.merge(qrels, on='qid', how='inner')
        qrels.to_csv(f'{output}/datasets.qrels/{d}.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'did', 'pid', 'rel'])
        qrels.drop_duplicates(inplace=True)
        qrels.to_csv(f'{output}/datasets.qrels/{d}.qrels.nodups.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'did', 'pid', 'rel'])
