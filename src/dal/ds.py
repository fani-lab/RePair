import json, pandas as pd
from tqdm import tqdm
from os.path import isfile,join

from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from ftfy import fix_text

class Dataset(object):
    searcher = None
    settings = None

    def __init__(self, settings):
        Dataset.settings = settings
        # https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
        #searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        # sometimes we need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
        # sometimes we need to manually build the index ==> Aol.init()
        Dataset.user_pairing = "user/" if "user" in settings["pairing"] else ""
        # index_item_str = '.'.join(settings["index_item"]) if self.__class__.__name__ != 'MsMarcoPsg' else ""
        # Dataset.searcher = LuceneSearcher(f'{Dataset.settings["index"]}{self.user_pairing}{index_item_str}')

        #Dataset.user_pairing = "label/" if "label" in settings["pairing"] else ""
        #index_item_str = '.'.join(settings["index_item"]) if self.__class__.__name__ != 'MsMarcoPsg' else ""
        Dataset.searcher = LuceneSearcher(Dataset.settings["index"]) #LuceneSearcher(f'{Dataset.settings["index"]}')

        if not Dataset.searcher: raise ValueError(f'Lucene searcher cannot find/build index at {Dataset.settings["index"]}!')

    @classmethod
    def _txt(cls, pid):
        # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
        # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
        try:return fix_text(json.loads(cls.searcher.doc(str(pid)).raw())['contents'].lower().replace('\n', '').replace(':', '').replace(
                '\r', '').replace(',', '').replace('\t', ''))
        except AttributeError: return ''  # if Dataset.searcher.doc(str(pid)) is None
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
    def search(cls, in_query, out_docids, qids, ranker='bm25', topk=100, batch=None, ncores=1, index=None):
        print(f'Searching docs for {in_query} ...')
        # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
        # with open(in_query, 'r', encoding='utf-8') as f: [to_docids(l) for l in f]
        queries = pd.read_csv(in_query, names=['query'], sep='\r\r', skip_blank_lines=False, engine='python')  # a query might be empty str (output of t5)!!
        assert len(queries) == len(qids)
        cls.search_df(queries, out_docids, qids, ranker=ranker, topk=topk, batch=batch, ncores=ncores, index=index)

    @classmethod
    def search_df(cls, queries, out_docids, qids, ranker='bm25', topk=100, batch=None, ncores=1, index=None,encoder=None):

        if not cls.searcher:
            if ranker == 'tct_colbert':
                cls.encoder = TctColBertQueryEncoder('castorini/tct_colbert-v2-hnp-msmarco')
                if 'msmarco.passage' in out_docids.split('/'):
                    cls.searcher = FaissSearcher.from_prebuilt_index(index, cls.encoder)
                if 'orcas' in out_docids.split('/'):
                    cls.searcher = FaissSearcher('../data/raw/toy.orcas/faiss-flat.msmarco-doc-per-passage.tct_colbert-v2-hnp', cls.encoder)
                    #cls.searcher = FaissSearcher.from_prebuilt_index('msmarco-doc-tct_colbert-v2-hnp-bf', cls.encoder)
                else:
                    cls.searcher = FaissSearcher(index, cls.encoder)
            else:
                cls.searcher = LuceneSearcher("../data/raw/toy.orcas/lucene-index.msmarco-v1-doc.20220131.9ea315")

        if ranker == 'bm25': cls.searcher.set_bm25(0.82, 0.68)
        if ranker == 'qld': cls.searcher.set_qld()
        with open(out_docids, 'w', encoding='utf-8') as o:
            if batch:
                for b in tqdm(range(0, len(queries), batch)):
                    # qids must be in list[str]!
                    hits = cls.searcher.batch_search(queries.iloc[b: b + batch]['query'].astype(str).values.tolist(), qids[b: b + batch], k=topk, threads=ncores)
                    for qid in hits.keys():
                        for i, h in enumerate(hits[qid]):  # hits are sorted desc based on score => required for trec_eval.9.0.4
                            o.write(f'{qid}\tQ0\t{h.docid:15}\t{i + 1:2}\t{h.score:.5f}\tPyserini Batch\n')
            else:
                def _docids(row):
                    if pd.isna(row.query): return  # in the batch call, they do the same. Also, for '', both return [] with no exception
                    if ranker == 'tct_colbert':
                        hits = cls.searcher.search(row.query, k=topk)
                        unique_docids = set()
                        for i, h in enumerate(hits):
                            if h.docid not in unique_docids:
                                unique_docids.add(h.docid)
                                o.write(f'{qids[row.name]}\tQ0\t{h.docid:7}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')
                        if len(unique_docids) < topk:
                            print(f'unique docids fetched less than {topk}')
                    else:
                        hits = cls.searcher.search(row.query, k=topk, remove_dups=True)
                        for i, h in enumerate(hits): o.write(f'{qids[row.name]}\tQ0\t{h.docid:7}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')

                queries.progress_apply(_docids, axis=1)

    @classmethod
    def aggregate(cls, original, changes, output, is_large_ds=False):
        ranker = changes[0][1].split('.')[4]  # e.g., pred.0-1004000.bm25.success.10 => bm25
        metric = '.'.join(changes[0][1].split('.')[5:-1])  # e.g., pred.0-1004000.bm25.success.10 => success.10
        for change, metric_value in changes:
            pred = pd.read_csv((join(output, change)+".tsv"), sep='\r\r', skip_blank_lines=False, names=[change], converters={change: cls.clean}, engine='python', index_col=False, header=None)
            assert len(original['qid']) == len(pred[change])
            if is_large_ds:
                pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1, 2], names=['qid', f'{change}.{ranker}.{metric}'], index_col=False, dtype={'qid': str})
            else:
                pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1, 2],engine='python', names=['qid', f'{change}.{ranker}.{metric}'], index_col=False,skipfooter=1, dtype={'qid': str})
            original[change] = pred  # to know the actual change
            original = original.merge(pred_metric_values, how='left', on='qid')  # to know the metric value of the change
            original[f'{change}.{ranker}.{metric}'].fillna(0, inplace=True)

        print(f'Saving original queries, all their changes, and their {metric} values based on {ranker} ...')
        original.to_csv(f'{output}/{ranker}.{metric}.agg.test.all_.tsv', sep='\t', encoding='UTF-8', index=False)

        print(f'Saving original queries, better changes, and {metric} values based on {ranker} ...')
        with open(f'{output}/{ranker}.{metric}.agg.gold.tsv', mode='w', encoding='UTF-8') as agg_gold, \
                open(f'{output}/{ranker}.{metric}.agg.all.tsv', mode='w', encoding='UTF-8') as agg_all, \
                open(f'{output}/{ranker}.{metric}.agg.plat.tsv', mode='w', encoding='UTF-8') as agg_plat:
            agg_gold.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
            agg_all.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
            agg_plat.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
            for index, row in tqdm(original.iterrows(), total=original.shape[0]):
                agg_gold.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                agg_plat.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                agg_all.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                all = list()
                for change, metric_value in changes: all.append((row[change], row[f'{change}.{ranker}.{metric}'], change))
                all = sorted(all, key=lambda x: x[1], reverse=True)
                for i, (query, metric_value, change) in enumerate(all):
                    agg_all.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')
                    if metric_value > 0 and metric_value >= row[f'original.{ranker}.{metric}']: agg_gold.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')
                    if metric_value > 0 and metric_value > row[f'original.{ranker}.{metric}']: agg_plat.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')

    @classmethod
    def box(cls, input, qrels, output, checks):
        ranker = input.columns[-1].split('.')[0]  # e.g., bm25.success.10 => bm25
        metric = '.'.join(input.columns[-1].split('.')[1:])  # e.g., bm25.success.10 => success.10
        for c in checks.keys():
            print(f'Boxing {c} queries for {ranker}.{metric} ...')
            ds = {'qid': list(), 'query': list(), f'{ranker}.{metric}': list(), 'query_': list(), f'{ranker}.{metric}_': list()}
            groups = input.groupby('qid')
            for _, group in tqdm(groups, total=len(groups)):
                if len(group) >= 2:
                    original_q, original_q_metric = group.iloc[0], group.iloc[0][f'{ranker}.{metric}']
                    refined_q, refined_q_metric = group.iloc[1], group.iloc[1][f'{ranker}.{metric}']
                    for i in range(1,2):  # len(group)): #IMPORTANT: We can have more than one golden query with SAME metric value. Here we skip them so the qid will NOT be replicated!
                        if (group.iloc[i][f'{ranker}.{metric}'] < refined_q[f'{ranker}.{metric}']): break
                        if not eval(checks[c]): break
                        ds['qid'].append(original_q['qid'])
                        ds['query'].append(original_q['query'])
                        ds[f'{ranker}.{metric}'].append(original_q_metric)
                        ds['query_'].append(group.iloc[i]['query'])
                        ds[f'{ranker}.{metric}_'].append(refined_q_metric)  # TODO: we can add golden queries with same metric value as a list here

            df = pd.DataFrame.from_dict(ds).astype({'qid':str})
            # df.drop_duplicates(subset=['qid'], inplace=True)
            del ds
            df.to_csv(f'{output}/{c}.tsv', sep='\t', encoding='utf-8', index=False, header=False)
            print(f'{c}  has {df.shape[0]} queries')
            qrels = df.merge(qrels, on='qid', how='inner')
            qrels.to_csv(f'{output}/{c}.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=['qid', 'did', 'pid', 'rel'])

    @classmethod
    def stat(cls, agg_all, output_path, raw_path, ranker, metric,dataset):pass


#dic_qid.append({group["qid"].values[0]:{"mean_diff":mean_diff,"max_diff":max_diff}})
#qid
#(group.iloc[1:len(group)])["bm25.map"].mean()
#group[group["bm25.map"]>0]
#group.iloc[1:len(group)][group["bm25.map"]>0]["bm25.map"].max()

#qid
#queries.loc[(queries['qid'] == int(group["qid"].values[0]))]["label"].values[0]