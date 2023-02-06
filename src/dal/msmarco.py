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
    if not isfile(f'{output}/bm25.map.agg.all.tsv'):
        print('no aggregate file found, Writing them all now!\n')
        for file, file_map in prediction_files_list:
            pred_df = pd.read_csv(join(output, file), converters={'query': to_norm}, sep='\r\r', skip_blank_lines=False,
                                  names=[f'{file}_query'], engine='python', index_col=False, header=None)
            assert len(original['qid']) == len(pred_df[f'{file}_query'])
            pred_df['qid'] = original['qid']
            pred_df_map = pd.read_csv(join(output, file_map), sep='\t', names=['map', 'qid', f'{file}_map'],
                                      index_col=False, low_memory=False)
            pred_df_map.drop(columns=['map'], inplace=True)
            pred_df_map.drop(pred_df_map.tail(1).index, inplace=True)
            original = original.merge(pred_df, how='left', on='qid')
            original = original.merge(pred_df_map, how='left', on='qid')

        print('saving all merged queries\n')
        original.to_csv(f'{output}/bm25.map.agg.all.tsv', sep='\t', encoding='UTF-8', index=False)
    else:
        print('retrieving saved file with all merged queries.\n')
        original = pd.read_csv(f'{output}/bm25.map.agg.all.tsv', sep='\t', encoding='UTF-8')
    print('calculating performance of predicted queries from aggregate file\n')
    with open(f'{output}/bm25.map.agg.best.tsv', mode='w', encoding='UTF-8') as agg_best:
        agg_best.write('qid\torder\tquery\tmap\n')
        original['og_map'] = original['og_map'].fillna(0)
        for i in range(1,25):
            original[f'pred.{i}-1004000_map'] = original[f'pred.{i}-1004000_map'].fillna(0)
        for index, row in tqdm(original.iterrows(), total=original.shape[0]):
            agg_best.write(f'{row.qid}\t-1\t{row.query}\t{row.og_map}\n')
            best_results = list()
            for i in range(1, 25):
                if row[f'pred.{i}-1004000_map'] >= row['og_map']:
                    best_results.append((row[f'pred.{i}-1004000_query'], row[f'pred.{i}-1004000_map'], i))
            best_results = sorted(best_results, key=lambda x: x[1], reverse=True)
            for i, (query, map_val, fileid) in enumerate(best_results): agg_best.write(f'{row.qid}\t{fileid}\t{query}\t{map_val}\n')
    print('saved file for all predicted queries that performed better than the original query\n')
    return 0

def create_dataset(input,qrels,output):
    diamond_dict = {'qid': list(), 'i_query': list(), 'i_map': list(), 'f_query': list(),
                    'f_map': list()}
    platinum_dict = {'qid': list(), 'i_query': list(), 'i_map': list(), 'f_query': list(),
                     'f_map': list()}
    gold_dict = {'qid': list(), 'i_query': list(), 'i_map': list(), 'f_query': list(),
                 'f_map': list()}
    print('creating diamond,platinum and gold dataset')
    for name, group in input.groupby('qid'):
        df = group.head(2)
        if df.shape[0] == 2:
            # diamond_queries
            if (df.iloc[1]['map'] == 1 and df.iloc[0]['map'] < 1):
                diamond_dict['qid'].append(df.iloc[0]['qid'])
                if (len(df.iloc[0][2]) < 7):
                    diamond_dict['i_query'].append(group.tail(1)['query'].values[0])
                    diamond_dict['i_map'].append(group.tail(1)['map'].values[0])
                else:
                    diamond_dict['i_query'].append(df.iloc[0]['query'])
                    diamond_dict['i_map'].append(df.iloc[0]['map'])

                diamond_dict['f_query'].append(df.iloc[1]['query'])
                diamond_dict['f_map'].append(df.iloc[1]['map'])
            # platinum queries
            if (df.iloc[1]['map'] > df.iloc[0]['map']):
                platinum_dict['qid'].append(df.iloc[0]['qid'])
                if (len(df.iloc[0][2]) < 7):
                    platinum_dict['i_query'].append(group.tail(1)['query'].values[0])
                    platinum_dict['i_map'].append(group.tail(1)['map'].values[0])
                else:
                    platinum_dict['i_query'].append(df.iloc[0]['query'])
                    platinum_dict['i_map'].append(df.iloc[0]['map'])
                platinum_dict['f_query'].append(df.iloc[1]['query'])
                platinum_dict['f_map'].append(df.iloc[1]['map'])
            # gold_queries
            if (df.iloc[1]['map'] >= df.iloc[0]['map']):
                gold_dict['qid'].append(df.iloc[0]['qid'])
                if (len(df.iloc[0][2]) < 7):
                    gold_dict['i_query'].append(group.tail(1)['query'].values[0])
                    gold_dict['i_map'].append(group.tail(1)['map'].values[0])
                else:
                    gold_dict['i_query'].append(df.iloc[0]['query'])
                    gold_dict['i_map'].append(df.iloc[0]['map'])
                gold_dict['f_query'].append(df.iloc[1]['query'])
                gold_dict['f_map'].append(df.iloc[1]['map'])
        else:
            if(df.iloc[0]['map'] == 1):
                diamond_dict['qid'].append(df.iloc[0]['qid'])
                diamond_dict['i_query'].append(df.iloc[0]['query'])
                diamond_dict['i_map'].append(df.iloc[0]['map'])
                diamond_dict['f_query'].append(df.iloc[0]['query'])
                diamond_dict['f_map'].append(df.iloc[0]['map'])
            if (df.iloc[0]['map'] > 0):
                platinum_dict['qid'].append(df.iloc[0]['qid'])
                platinum_dict['i_query'].append(df.iloc[0]['query'])
                platinum_dict['i_map'].append(df.iloc[0]['map'])
                platinum_dict['f_query'].append(df.iloc[0]['query'])
                platinum_dict['f_map'].append(df.iloc[0]['map'])
            gold_dict['qid'].append(df.iloc[0]['qid'])
            gold_dict['i_query'].append(df.iloc[0]['query'])
            gold_dict['i_map'].append(df.iloc[0]['map'])
            gold_dict['f_query'].append(df.iloc[0]['query'])
            gold_dict['f_map'].append(df.iloc[0]['map'])

    diamond = pd.DataFrame.from_dict(diamond_dict)
    diamond.to_csv(f'{output}/queries/diamond.tsv', sep='\t', encoding='utf-8', index=False, header=False)
    diamond.to_csv(f'{output}/queries/diamond_initial.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'i_query'])
    diamond.to_csv(f'{output}/queries/diamond_target.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                   columns=['qid', 'f_query'])
    print('saved diamond queries')
    print(f'diamond dataset has {diamond.shape[0]} queries with map 1')
    diamond_qrels = diamond.merge(qrels, on='qid', how='inner')
    diamond_qrels.to_csv(f'{output}/qrels/diamond.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                         columns=['qid', 'did', 'pid', 'rel'])
    diamond_qrels.drop_duplicates(inplace=True)
    diamond_qrels.to_csv(f'{output}/qrels/diamond.qrels.nodups.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                         columns=['qid', 'did', 'pid', 'rel'])
    print('saved diamond qrels')
    platinum = pd.DataFrame.from_dict(platinum_dict)
    platinum.to_csv(f'{output}/queries/platinum.tsv', sep='\t', encoding='utf-8', index=False, header=False)
    platinum.to_csv(f'{output}/queries/platinum_initial.tsv', sep='\t', encoding='utf-8', index=False, header=False,columns=['qid', 'i_query'])
    platinum.to_csv(f'{output}/queries/platinum_target.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                    columns=['qid', 'f_query'])
    print('saved platinum queries')
    print(f'platinum dataset has {platinum.shape[0]} queries with map greater than the original query')
    platinum_qrels = platinum.merge(qrels, on='qid', how='inner')
    platinum_qrels.to_csv(f'{output}/qrels/platinum.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                         columns=['qid', 'did', 'pid', 'rel'])
    platinum_qrels.drop_duplicates(inplace=True)
    platinum_qrels.to_csv(f'{output}/qrels/platinum.qrels.nodups.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                         columns=['qid', 'did', 'pid', 'rel'])
    print('saved platinum qrels')
    gold = pd.DataFrame.from_dict(gold_dict)
    gold.to_csv(f'{output}/queries/gold.tsv', sep='\t', encoding='utf-8', index=False, header=False)
    gold.to_csv(f'{output}/queries/gold_initial.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid','i_query'])
    gold.to_csv(f'{output}/queries/gold_target.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'f_query'])
    print('saved gold queries')
    print(f'gold dataset has {gold.shape[0]} queries with map greater than or equal to original query')
    gold_qrels = gold.merge(qrels, on='qid',how='inner')
    gold_qrels.to_csv(f'{output}/qrels/gold.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                         columns=['qid', 'did', 'pid', 'rel'])
    gold_qrels.drop_duplicates(inplace=True)
    gold_qrels.to_csv(f'{output}/qrels/gold.qrels.nodups.tsv', sep='\t', encoding='utf-8', index=False, header=False,
                         columns=['qid', 'did', 'pid', 'rel'])
    print('saved gold qrels')
