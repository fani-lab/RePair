import csv
from tqdm import tqdm
from os.path import join
import json, pandas as pd
from src.dal.query import Query
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder


class Dataset(object):
    queries = []
    searcher = None
    settings = None

    def __init__(self, settings, domain):
        Dataset.settings = settings
        # https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
        # searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        # sometimes we need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
        # sometimes we need to manually build the index ==> Aol.init()
        Dataset.user_pairing = "user/" if "user" in settings["pairing"] else ""
        index_item_str = '.'.join(settings["index_item"]) if self.__class__.__name__ != 'MsMarcoPsg' else ""
        Dataset.searcher = LuceneSearcher(f'{Dataset.settings["index"]}{self.user_pairing}{index_item_str}')
        Dataset.domain = domain

        if not Dataset.searcher: raise ValueError(f'Lucene searcher cannot find/build index at {Dataset.settings["index"]}!')

    @classmethod
    def read_queries(cls, input, domain, trec=False):
        is_tag_file = False
        q, qid = '', ''
        queries = pd.DataFrame(columns=['qid'])
        with open(f'{input}/topics.{domain}.txt', 'r', encoding='UTF-8') as Qfile:
            for line in Qfile:
                if '<top>' in line and not is_tag_file: is_tag_file = True
                if '<num>' in line:
                    qid = str(line[line.index(':') + 1:])
                elif line[:7] == '<title>':
                    q = line[8:].strip()
                    if not q: q = next(Qfile).strip()
                elif '<topic' in line:
                    s = line.index('\"') + 1
                    e = line.index('\"', s + 1)
                    qid = str(line[s:e])
                elif line[2:9] == '<query>':
                    q = line[9:-9]
                elif len(line.split('\t')) >= 2 and not is_tag_file:
                    qid = str(line.split('\t')[0].rstrip())
                    q = line.split('\t')[1].rstrip()
                if q != '' and qid != '':
                    new_line = {'qid': qid, 'query': q}
                    queries = pd.concat([queries, pd.DataFrame([new_line])], ignore_index=True)
                    q, qid = '', ''
        infile = f'{input}/qrels.{domain}.txt'
        with open(infile, 'r') as file:
            line = file.readline()
            if '\t' in line: separator = '\t'
            elif '  ' in  line: separator = '  '
            else: separator = '\s'
        qrels = pd.read_csv(infile, sep=separator, index_col=False, names=['qid', '0', 'did', 'relevancy'], header=None, engine='python', encoding='unicode_escape', dtype={'qid': str})
        qrels.drop_duplicates(subset=['qid', 'did'], inplace=True)  # qrels have duplicates!!
        if trec: qrels.to_csv(f'{input}/qrels.{domain}.train.tsv_', index=False, sep='\t', header=False)
        else: qrels.to_csv(f'{input}/qrels.train.tsv_', index=False, sep='\t', header=False)
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
        queries_qrels = queries_qrels.sort_values(by='qid')
        cls.create_query_objects(queries_qrels, ['qid', '0', 'did', 'relevancy'], domain)

    @classmethod
    def pair(cls, input, output, index_item, cat=True): pass

    @classmethod
    def _txt(cls, pid):
        # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
        # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
        try: return json.loads(cls.searcher.doc(str(pid)).raw())['contents'].lower()
        except AttributeError: return ''  # if Dataset.searcher.doc(str(pid)) is None
        except Exception as e: raise e

    @classmethod
    def create_query_objects(cls, queries_qrels, qrel_col, domain):
        qid = ""
        query = None
        for i, row in queries_qrels.iterrows():
            if qid != row['qid']:
                if query: cls.queries.append(query)
                qid = row['qid']
                query = Query(domain=domain, qid=qid, q=row['query'], qrel={col: [] for col in qrel_col})
            [query.qrel[col].append(str(row[col])) for col in qrel_col]
            query.original = True
        if query: cls.queries.append(query)

    # gpu-based t5 generate the predictions in b'' format!!!
    @classmethod
    def clean(cls, tf_txt):
        # lambda x: x.replace('b\'', '').replace('\'', '') if in pandas' convertors
        # TODO: we need to clean the \\x chars also
        return tf_txt.replace('b\'', '').replace('\'', '').replace('b\"', '').replace('\"', '')

    @classmethod
    def search(cls, in_query:str, out_docids:str, qids:list, ranker='bm25', topk=100, batch=None, ncores=1, index=None):
        ansi_reset = "\033[0m"
        print(f'Searching docs for {hex_to_ansi("#3498DB")}{in_query} {ansi_reset}and writing results in {hex_to_ansi("#F1C40F")}{out_docids}{ansi_reset} ...')
        # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
        # with open(in_query, 'r', encoding='utf-8') as f: [to_docids(l) for l in f]
        if (in_query.split('/')[-1]).split('.')[0] == 'refiner': queries = pd.read_csv(in_query, names=['query'], sep='\t', usecols=[1], skip_blank_lines=False, engine='python')
        else: queries = pd.read_csv(in_query, names=['query'], sep='\r\r', skip_blank_lines=False, engine='python')  # a query might be empty str (output of t5)!!
        assert len(queries) == len(qids)
        cls.search_df(queries, out_docids, qids, ranker=ranker, topk=topk, batch=batch, ncores=ncores, index=index)

    @classmethod
    def search_df(cls, queries, out_docids, qids, ranker='bm25', topk=100, batch=None, ncores=1, index=None, encoder=None):
        if not cls.searcher:
            if ranker == 'tct_colbert':
                cls.encoder = TctColBertQueryEncoder(encoder)
                if 'msmarco.passage' in out_docids.split('/'): cls.searcher = FaissSearcher.from_prebuilt_index(index, cls.encoder)
                else: cls.searcher = FaissSearcher(index, cls.encoder)
            else: cls.searcher = LuceneSearcher(index)

        if ranker == 'bm25': cls.searcher.set_bm25(0.82, 0.68)
        if ranker == 'qld': cls.searcher.set_qld()
        with open(out_docids, 'w', encoding='utf-8') as o:
            if batch:
                for b in tqdm(range(0, len(queries), batch)):
                    # qids must be in list[str]!
                    hits = cls.searcher.batch_search(queries.iloc[b: b + batch]['query'].values.tolist(), qids[b: b + batch], k=topk, threads=ncores)
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
                        #TODO: Store qrets
                        hits = cls.searcher.search(row.query, k=topk, remove_dups=True)
                        for i, h in enumerate(hits): o.write(f'{qids[row.name]}\tQ0\t{h.docid:7}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')

                queries.apply(_docids, axis=1)

    @classmethod
    def aggregate(cls, original, changes, output, is_large_ds=False):
        ranker = changes[0][1].split('.')[2]  # e.g., pred.0-1004000.bm25.success.10 => bm25
        metric = '.'.join(changes[0][1].split('.')[3:])  # e.g., pred.0-1004000.bm25.success.10 => success.10

        for change, metric_value in changes:
            if 'refiner.' in change: pred = pd.read_csv(join(output, change), sep='\t', usecols=[1], skip_blank_lines=False, names=[change], converters={change: cls.clean}, engine='python', index_col=False, header=None)
            else: pred = pd.read_csv(join(output, change), sep='\r\r', skip_blank_lines=False, names=[change], converters={change: cls.clean}, engine='python', index_col=False, header=None)
            assert len(original['qid']) == len(pred[change])
            if is_large_ds:
                pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1, 2], names=['qid', f'{change}.{ranker}.{metric}'], index_col=False, dtype={'qid': str})
            else:
                pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1, 2], names=['qid', f'{change}.{ranker}.{metric}'], index_col=False,skipfooter=1, dtype={'qid': str}, engine='python')
            original[change] = pred  # to know the actual change
            original = original.merge(pred_metric_values, how='left', on='qid')  # to know the metric value of the change
            original[f'{change}.{ranker}.{metric}'].fillna(0, inplace=True)

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
            print(f'{c}  has {df.shape[0]} queries\n')
            df = df.merge(qrels, on='qid', how='inner')
            df.to_csv(f'{output}/{c}.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=list(qrels.columns))


def hex_to_ansi(hex_color_code):
    hex_color_code = hex_color_code.lstrip('#')
    red = int(hex_color_code[0:2], 16)
    green = int(hex_color_code[2:4], 16)
    blue = int(hex_color_code[4:6], 16)
    return f'\033[38;2;{red};{green};{blue}m'
