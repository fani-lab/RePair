import json, pandas as pd
from tqdm import tqdm
from os.path import isfile,join

from pyserini.search.lucene import LuceneSearcher

class Dataset(object):
    searcher = None
    settings = None

    def __init__(self, settings):
        Dataset.settings = settings
        # https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
        # searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        # sometimes we need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
        # sometimes we need to manually build the index ==> Aol.init()
        Dataset.searcher = LuceneSearcher(Dataset.settings['index'])
        if not Dataset.searcher: raise ValueError(f'Lucene searcher cannot find/build msmarco.passage index at {Dataset.settings["index"]}!')

    @classmethod
    def init(cls, homedir, index_item, indexdir, ncore): pass

    @classmethod
    def _txt(cls, pid):
        # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
        # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
        try:return json.loads(Dataset.searcher.doc(str(pid)).raw())['contents'].lower()
        except AttributeError: return '' #if Dataset.searcher.doc(str(pid)) is None
        except Exception as e: raise e

    @classmethod
    def pair(cls, input, output, index_item, cat=True): pass

    # gpu-based t5 generate the predictions in b'' format!!!
    @classmethod
    def clean(cls, tf_txt):
        # lambda x: x.replace('b\'', '').replace('\'', '') if in pandas' convertors
        # TODO: we need to clean the \\x chars also
        return tf_txt.replace('b\'', '').replace('\'', '').replace('b\"', '').replace('\"', '')

    @classmethod
    def search(cls, in_query, out_docids, qids, ranker='bm25', topk=100, batch=None, ncores=1):
        print(f'Searching docs for {in_query} ...')
        # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
        # with open(in_query, 'r', encoding='utf-8') as f: [to_docids(l) for l in f]
        queries = pd.read_csv(in_query, names=['query'], sep='\r', skip_blank_lines=False, engine='c')  # a query might be empty str (output of t5)!!
        assert len(queries) == len(qids)
        cls.search_df(queries, out_docids, qids, ranker=ranker, topk=topk, batch=batch, ncores=ncores)

    @classmethod
    def search_df(cls, queries, out_docids, qids, ranker='bm25', topk=100, batch=None, ncores=1):
        if ranker == 'bm25': Dataset.searcher.set_bm25(0.82, 0.68)
        if ranker == 'qld': Dataset.searcher.set_qld()
        with open(out_docids, 'w', encoding='utf-8') as o:
            if batch:
                for b in tqdm(range(0, len(queries), batch)):
                    # qids must be in list[str]!
                    hits = Dataset.searcher.batch_search(queries.iloc[b: b + batch]['query'].values.tolist(), qids[b: b + batch], k=topk, threads=ncores)
                    for qid in hits.keys():
                        for i, h in enumerate(hits[qid]):  # hits are sorted desc based on score => required for trec_eval
                            o.write(f'{qid}\tQ0\t{h.docid:15}\t{i + 1:2}\t{h.score:.5f}\tPyserini Batch\n')
            else:
                def _docids(row):
                    if pd.isna(row.query): return  # in the batch call, they do the same. Also, for '', both return [] with no exception
                    hits = Dataset.searcher.search(row.query, k=topk, remove_dups=True)
                    for i, h in enumerate(hits): o.write(f'{qids[row.name]}\tQ0\t{h.docid:7}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')

                queries.progress_apply(_docids, axis=1)

    @staticmethod
    def aggregate(original, changes, output):
        ranker = changes[0][1].split('.')[2]  # e.g., pred.0-1004000.bm25.success.10 => bm25
        metric = '.'.join(changes[0][1].split('.')[3:])  # e.g., pred.0-1004000.bm25.success.10 => success.10

        for change, metric_value in changes:
            pred = pd.read_csv(join(output, change), sep='\r', skip_blank_lines=False, names=[change], engine='c', index_col=False, header=None)
            assert len(original['qid']) == len(pred[change])
            pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1, 2], names=['qid', f'{change}.{ranker}.{metric}'], index_col=False, skipfooter=1)
            original[change] = pred  # to know the actual change
            original = original.merge(pred_metric_values, how='left', on='qid')  # to know the metric value of the change
            # original[f'{change}.{ranker}.{metric}'].fillna(0, inplace=True)

        print(f'Saving original queries, all their changes, and their {metric} values based on {ranker} ...')
        original.to_csv(f'{output}/{ranker}.{metric}.agg.all_.tsv', sep='\t', encoding='UTF-8', index=False)

        print(f'Saving original queries, better changes, and {metric} values based on {ranker} ...')
        with open(f'{output}/{ranker}.{metric}.agg.gold.tsv', mode='w', encoding='UTF-8') as agg_gold, \
                open(f'{output}/{ranker}.{metric}.agg.all.tsv', mode='w', encoding='UTF-8') as agg_all:
            agg_gold.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
            agg_all.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
            for index, row in tqdm(original.iterrows(), total=original.shape[0]):
                agg_gold.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                agg_all.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                all = list()
                for change, metric_value in changes: all.append((row[change], row[f'{change}.{ranker}.{metric}'], change))
                all = sorted(all, key=lambda x: x[1], reverse=True)
                for i, (query, metric_value, change) in enumerate(all):
                    agg_all.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')
                    if metric_value > 0 and metric_value >= row[f'original.{ranker}.{metric}']: agg_gold.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')


    @staticmethod
    def box(input, qrels, output, checks):
        ranker = input.columns[-1].split('.')[0]  # e.g., bm25.success.10 => bm25
        metric = '.'.join(input.columns[-1].split('.')[1:])  # e.g., bm25.success.10 => success.10
        for c in checks.keys():
            print(f'Boxing {c} queries for {ranker}.{metric} ...')
            ds = {'qid': list(), 'query': list(), f'{ranker}.{metric}': list(), 'query_': list(), f'{ranker}.{metric}_': list()}
            groups = input.groupby('qid')
            for _, group in tqdm(groups, total=len(groups)):
                if len(group) >= 2:
                    original_q, original_q_metric = group.iloc[0], group.iloc[0][f'{ranker}.{metric}']
                    changed_q, changed_q_metric = group.iloc[1], group.iloc[1][f'{ranker}.{metric}']
                    for i in range(1,2):  # len(group)): #IMPORTANT: We can have more than one golden query with SAME metric value. Here we skip them so the qid will NOT be replicated!
                        if (group.iloc[i][f'{ranker}.{metric}'] < changed_q[f'{ranker}.{metric}']): break
                        if not eval(checks[c]): break  # for gold this is always true since we put >= metric values in *.agg.best.tsv
                        ds['qid'].append(original_q['qid'])
                        ds['query'].append(original_q['query'])
                        ds[f'{ranker}.{metric}'].append(original_q_metric)
                        ds['query_'].append(group.iloc[i]['query'])
                        ds[f'{ranker}.{metric}_'].append(changed_q_metric)  # TODO: we can add golden queries with same metric value as a list here

            df = pd.DataFrame.from_dict(ds).astype({'qid':str})
            # df.drop_duplicates(subset=['qid'], inplace=True)
            del ds
            df.to_csv(f'{output}/{c}.tsv', sep='\t', encoding='utf-8', index=False, header=False)
            print(f'{c}  has {df.shape[0]} queries')
            qrels = df.merge(qrels, on='qid', how='inner')
            qrels.to_csv(f'{output}/{c}.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'did', 'pid', 'rel'])





