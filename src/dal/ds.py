import re
import os
import math
import numpy as np
from itertools import product, combinations
from tqdm import tqdm
from os.path import join
import json, pandas as pd
from ftfy import fix_text
from cmn.query import Query
from refinement.refiner_param import refiners
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder


class Dataset(object):
    domain = None
    queries = None
    index = None
    searcher = None
    settings = None

    def __init__(self, settings, domain):
        # https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
        # searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        # sometimes we need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
        # sometimes we need to manually build the index ==> Aol.init()
        Dataset.domain = domain
        Dataset.queries = []
        Dataset.queries_qrels = pd.DataFrame({})
        Dataset.index = ""
        Dataset.searcher = None
        Dataset.settings = settings

    @classmethod
    def read_queries(cls, input, domain, trec=False):
        is_tag_file = False
        q, qid = '', ''
        queries = pd.DataFrame(columns=['qid'])
        with open(f'{input}/topics.{domain}.txt', 'r', encoding='UTF-8') as Qfile:
            for line in Qfile:
                if '<top>' in line and not is_tag_file: is_tag_file = True
                if '<num>' in line:
                    qid = re.findall(r'\d+', line)[0]
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
        cls.queries_qrels = pd.concat([cls.queries_qrels, queries], axis=0)
        # cls.create_query_objects(queries_qrels, ['qid', '0', 'did', 'relevancy'], domain)

    @classmethod
    def pairing(cls, input):
        queries = cls.queries_qrels
        cls.index = cls.settings["index"]
        cls.searcher = LuceneSearcher(cls.index)
        docs_content = dict()
        # for index, row in qrels.iterrows(): docs_content[row['did']] = cls._txt(row['did'], input.split('/')[4])
        for ranker, metric in product(['bm25'], ['map']):
            print(f'Pairing results for {ranker}.{metric} ...')
            for category in ['all', 'global', 'local', 'bt_nllb', 'bt']:
                print(f'Pairing results for {category} ...')
                doc_list = pd.read_csv(f'../output/{input.split("/")[4]}/{ranker}.{metric}/rag/rag.{category}.k{60}.{ranker}', sep='\t', usecols=[0, 2], names=['qid', 'did'], index_col=False, skipfooter=1, dtype={'qid': str, 'did': str}, engine='python')
                doc_list['did'] = doc_list['did'].str.strip()

                # doc_list = pd.read_csv(f'../output/{input.split("/")[4]}/{ranker}.{metric}/rag/rag.{category}.k{60}.{ranker}', sep=' ', usecols=[0, 1], names=['qid', 'did'], index_col=False, skipfooter=1, dtype={'qid': str}, engine='python')
                # doc_list['did'] = doc_list['did'].str.replace(r'^Q0\t', '', regex=True).str.strip()

                queries_qrels = pd.merge(queries, doc_list, on='qid', how='inner', copy=False)
                queries_qrels = queries_qrels.sort_values(by='qid')
                cls.write_queries(queries_qrels, input.split('/')[4], 4, ranker, metric, category, docs_content)


    @classmethod
    def write_queries(cls, queries_qrels, domain, chunks, ranker, metric, category, docs_content):
        groups = queries_qrels.groupby(['qid', 'query'])

        output = f'../output/{domain}/{ranker}.{metric}/rag/initial_t5'
        if not os.path.isdir(output): os.makedirs(output)

        file_paths_qrel = [f'../output/{domain}/qrels{f".{i}" if chunks!=0 else ""}.tsv' for i in range(0, chunks)]
        file_paths_pairing = [f'{output}/pairing.{category}{f".{i}" if chunks!=0 else ""}.tsv' for i in range(0, chunks)]
        file_paths_original_qid_query = [f'{output}/original_qid.{category}{f".{i}" if chunks!=0 else ""}.tsv' for i in range(0, chunks)]
        file_paths_original_query = [f'{output}/original.{category}{f".{i}" if chunks!=0 else ""}.tsv' for i in range(0, chunks)]

        group_size = len(groups)//chunks
        for i, (name, group) in enumerate(groups):
            file_index = min(i // group_size, chunks-1)
            print(f'{name[0]}: Writing results to ../output/{domain}/{ranker}.{metric}/rag/initial_t5, index {file_index} ...')
            with open(file_paths_qrel[file_index], 'a', encoding='utf-8') as qrel, \
                 open(file_paths_pairing[file_index], 'a', encoding='utf-8') as pairing, \
                 open(file_paths_original_qid_query[file_index], 'a', encoding='utf-8') as original_qid, \
                 open(file_paths_original_query[file_index], 'a', encoding='utf-8') as original:
                query = name[1].replace("\t", " ").lower()

                docs = []
                # for index, row in group.iterrows(): docs.append((cls._txt(row['did'], domain)))
                for index, row in group.iterrows():
                    if row['did'] in docs_content:
                        docs.append(docs_content[row['did']])
                    else: docs.append((cls._txt(row['did'], domain)))
                doc = ' '.join(docs)
                doc = doc.encode('utf-8')
                qrel.write(f'{name[0]}\t{row["0"]}\t{row["did"]}\t{row["relevancy"]}\n')
                pairing.write(f'{query}\t{doc}\n')
                original_qid.write(f'{name[0]}\t{query}\n')
                original.write(f'{query}\n')


    @classmethod
    def _txt(cls, pid, domain):
        # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
        # if it is of type ``int``, it is treated as an internal Lucene``docid``. # stupid!!
        try:
            text = (cls.searcher.doc(str(pid)).raw()).lower()
            if domain == 'antique': return fix_text(json.loads(cls.searcher.doc(str(pid)).raw())['contents'].lower().replace('\n', '').replace(':', '').replace('\r', '').replace(',', '').replace('\t', ''))
            elif domain == 'robust04':
                start_tags = ['[text]', '<text>']
                end_tag = '</text>'
            elif domain == 'dbpedia':
                start_tags = ['<!-- rc -->\n<p>']
                end_tag = '</p>\n<!-- rc -->'
            elif domain == 'gov2':
                start_tags = ['<meta name="description"\ncontent="']
                end_tag = '">'

            while True:
                start_index = -1
                start_tag = None
                for tag in start_tags:
                    idx = text.find(tag)
                    if idx != -1 and (start_index == -1 or idx < start_index):
                        start_index = idx
                        start_tag = tag
                if start_index == -1: break
                end_index = text.find(end_tag, start_index)
                if end_index == -1: break
                text = text[start_index + len(start_tag): end_index]
            text = text.translate(str.maketrans('', '', "<>[]/\\"))
            return fix_text(text.replace('\n', '').replace(':','').replace('\r', '').replace(',', '').replace('\t', ''))
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
        if query: cls.queries.append(query)

    # gpu-based t5 generate the predictions in b'' format!!!
    @classmethod
    def clean(cls, tf_txt):
        # TODO: we need to clean the \\x chars also
        return tf_txt.replace('b\'', '').replace('\'', '').replace('b\"', '').replace('\"', '')

    @classmethod
    def set_index(cls, index): cls.index = index

    @classmethod
    def search(cls, in_query, out_docids:str, qids:list=[], ranker='bm25', topk=100, batch=None, ncores=1, encoder=None, index=None, settings=None):
        print(f'Searching docs for {hex_to_ansi("#3498DB")}{in_query}{hex_to_ansi(reset=True)} and writing results in {hex_to_ansi("#F1C40F")}{out_docids}{hex_to_ansi(reset=True)} ...')
        # https://github.com/google-research/text-to-text-transfer-transformer/issues/322
        # Initialization - All the variables in class cannot be shared!
        if isinstance(in_query, str):
            # if (in_query.split('/')[-1]).split('.')[0] == 'refiner' or in_query.split('/')[-1] == 'original': queries = pd.read_csv(in_query, names=['query'], sep='\t', usecols=[2], skip_blank_lines=False, engine='python')
            if (in_query.split('/')[-1]).split('.')[0] == 'super' or in_query.split('/')[-1] == 'original': queries = pd.read_csv(in_query, names=['query'], sep='\t', usecols=[1], skip_blank_lines=False, engine='python')
            else: queries = pd.read_csv(in_query, names=['query'], sep='\r\r', skip_blank_lines=False, engine='python')  # a query might be empty str (output of t5)!!
        else: queries = in_query
        qids = (pd.read_csv(in_query, names=['qid'], sep='\t', usecols=[0], skip_blank_lines=False, engine='python'))['qid'].tolist()

        assert len(queries) == len(qids)
        if not cls.index: cls.set_index(index)
        if not cls.settings: cls.settings = settings
        try:
            if not cls.searcher:
                # Dense Retrieval
                if ranker == 'tct_colbert':
                    cls.encoder = TctColBertQueryEncoder(encoder)
                    if 'msmarco.passage' in out_docids.split('/'): cls.searcher = FaissSearcher.from_prebuilt_index(cls.index, cls.encoder)
                    else: cls.searcher = FaissSearcher(cls.index, cls.encoder)
                # Sparce Retrieval
                else:
                    cls.searcher = LuceneSearcher(cls.index)
                    if ranker == 'bm25': cls.searcher.set_bm25(0.82, 0.68)
                    if ranker == 'qld': cls.searcher.set_qld()
        except Exception as e: raise ValueError(f'Lucene searcher cannot find/build index at {cls.settings["index"]}!')

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
    def get_refiner_list(cls, category):
        names = []
        if category == 'global' or category == 'local':
            names = list(refiners[category].keys())
        elif category == 'all':
            names = list(refiners['global'].keys()) + list(refiners['local'].keys())
        elif category == 'bt':
            names = [category]
        elif category == 'bt_nllb':
            names = [category]
        return [re.sub(r'\b(\w+)Stemmer\b', r'stem.\1', re.sub(r'\b\w*BackTranslation\w*\b', 'bt', name)).lower() for name in names]

    ''' Fuse Ranking '''
    @classmethod
    def reciprocal_rank_fusion(cls, docs, k, columns, output):
        docs = docs.groupby(['qid', 'did'])
        doc_fusion = pd.DataFrame(columns=columns)
        for group_name, group_df in docs:
            score = 0
            for index, row in group_df.iterrows():
                score += 1/(k+row['rank'])
            doc_fusion.loc[len(doc_fusion)] = [group_name[0], 'Q0', group_name[1], 0, score, 'Pyserini']
        doc_fusion = doc_fusion.sort_values(by=['qid', 'score'], ascending=[True, False])
        doc_fusion['rank'] = doc_fusion.groupby('qid').cumcount() + 1
        doc_fusion.to_csv(output, sep='\t', encoding='UTF-8', index=False, header=False)

    @classmethod
    def condorcet_fusion(cls, docs, columns, output):
        did_list = docs['did'].unique().tolist()
        docs = docs.groupby(['qid', 'refiner'])
        score_metrix = {did: 0 for did in did_list}
        pairwise_list = list(combinations(did_list, 2))

        doc_fusion = pd.DataFrame(columns=columns)
        for group_name, group_df in docs:
            for (did1, did2) in pairwise_list:
                r1 = (group_df.loc[group_df['did'] == did1, 'rank'].values)[0]
                r2 = (group_df.loc[group_df['did'] == did2, 'rank'].values)[0]
                if r1 > r2: score_metrix[did1] += 1
                else: score_metrix[did2] += 1
            for did in score_metrix.keys():
                doc_fusion.loc[len(doc_fusion)] = [group_name[0], 'Q0', did, 0, score_metrix[did], 'Pyserini']
                score_metrix[did] = 0

        doc_fusion = doc_fusion.sort_values(by=['id', 'score'], ascending=[True, False])
        doc_fusion['rank'] = doc_fusion.groupby('id').cumcount() + 1
        doc_fusion.to_csv(output, sep='\t', encoding='UTF-8', index=False, header=False)

    @classmethod
    def random(cls, docs, columns, output):
        docs = docs.groupby(['qid'])
        doc_fusion = pd.DataFrame(columns=columns)

        for qid, group_df in docs:
            df = group_df.drop_duplicates(subset='did', keep='first')
            df = df.sample(frac=1).reset_index(drop=True)
            df['score'] = np.random.rand(len(df))
            doc_fusion = pd.concat([doc_fusion, df], ignore_index=True)

        doc_fusion['rank'] = doc_fusion.groupby('id').cumcount() + 1
        doc_fusion.to_csv(output, sep='\t', encoding='UTF-8', index=False, header=False)


    # Example usage:
    # reciprocal_rank_fusion(cls, docs, [1, 2, 3], columns, 'output')
    def reciprocal_rank_fusion_multi_k(cls, docs, k_list, columns, output):
        docs = docs.groupby(['qid', 'did'])

        doc_fusion_dict = {k: pd.DataFrame(columns=columns) for k in k_list}

        for group_name, group_df in docs:
            scores = {k: 0 for k in k_list}

            for index, row in group_df.iterrows():
                for k in k_list:
                    scores[k] += 1 / (k + row['rank'])

            for k in k_list:
                doc_fusion_dict[k].loc[len(doc_fusion_dict[k])] = [group_name[0], 'Q0', group_name[1], 0, scores[k], 'Pyserini']
        for k, doc_fusion in doc_fusion_dict.items():
            doc_fusion = doc_fusion.sort_values(by=['qid', 'score'], ascending=[True, False])
            doc_fusion['rank'] = doc_fusion.groupby('qid').cumcount() + 1
            output_filename = output.split('.k.')[0] + '.k' + str(k) + '.' + output.split('.k.')[1]
            doc_fusion.to_csv(output_filename, sep='\t', encoding='UTF-8', index=False, header=False)


    @classmethod
    def aggregate(cls, original, refined_query, output, ranker, metric, selected_refiner='allref', cmd='agg'):
        def select(ref, change):
            if change.startswith('rag') and 'rag' in cmd:
                if ref in change: return True
                else: return False

            elif change.startswith('refiner') and not 'rag' in cmd:
                if ref == 'nllb' and not ref in change: return False        # only backtranslation with nllb
                elif ref == '-bt' and 'bt' in change: return False          # other refiners than backtranslartion
                elif ref == '+bt' and 'bt_bing' in change: return False     # all the refiners except bing
                else: return True                                           # all the refiners

            else: return False

        changes = [(f.split(f'.{ranker}.{metric}')[0], f) for f in os.listdir(output) if f.endswith(f'{ranker}.{metric}') and select(selected_refiner, f)]

        refiners = []
        for change, metric_value in changes:
            if 'rag' in cmd: refined = pd.DataFrame({change: [change] * len(original['qid'])})
            else: refined = pd.read_csv(f'{refined_query}/{change}', sep='\t', usecols=[2], skip_blank_lines=False, names=[change], converters={change: cls.clean}, engine='python', index_col=False, header=None)
            assert len(original['qid']) == len(refined[change])
            refiners.append(change)
            pred_metric_values = pd.read_csv(join(output, metric_value), sep='\t', usecols=[1, 2], names=['qid', f'{change}.{ranker}.{metric}'], index_col=False, skipfooter=1, dtype={'qid': str}, engine='python')
            original[change] = refined  # to know the actual change
            original = original.merge(pred_metric_values, how='left', on='qid')  # to know the metric value of the change
            original[f'{change}.{ranker}.{metric}'].fillna(0, inplace=True)

        print(f'Saving original queries, all their changes, and their {metric} values based on {ranker} ...')
        original.to_csv(f'{output}/{ranker}.{metric}.agg.{"all" if selected_refiner=="refiner" else selected_refiner}.tsv', encoding='UTF-8', index=False)
        if 'build' in cmd:
            output = f'../output/supervised/{(output.split("/"))[2]}'
            if not os.path.isdir(output): os.makedirs(output)
            cls.build(original, refiners, ranker, metric, f"{output}/{ranker}.{metric}.dataset.{'all' if selected_refiner=='refiner' else selected_refiner}.csv")
        else:
            output = f'{output}/agg'
            if cmd == 'rag' and selected_refiner == 'all': selected_refiner = 'allref'
            if not os.path.isdir(output): os.makedirs(output)
            print(f'Saving original queries, better changes, and {metric} values based on {ranker} ...')
            with open(f'{output}/{ranker}.{metric}.agg.{selected_refiner}.all.tsv', mode='w', encoding='UTF-8') as agg_all, \
                 open(f'{output}/{ranker}.{metric}.agg.{selected_refiner}.gold.tsv', mode='w', encoding='UTF-8') as agg_gold, \
                 open(f'{output}/{ranker}.{metric}.agg.{selected_refiner}.platinum.tsv', mode='w', encoding='UTF-8') as agg_plat, \
                 open(f'{output}/{ranker}.{metric}.agg.{selected_refiner}.negative.tsv', mode='w', encoding='UTF-8') as neg_exp:
                agg_all.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
                agg_gold.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
                agg_plat.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
                neg_exp.write(f'qid\torder\tquery\t{ranker}.{metric}\n')
                for index, row in tqdm(original.iterrows(), total=original.shape[0]):
                    agg_all.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                    agg_gold.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                    agg_plat.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                    neg_exp.write(f'{row.qid}\t-1\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\n')
                    all = list()
                    for change, metric_value in changes:
                        all.append((row[change], row[f'{change}.{ranker}.{metric}'], change))
                    all = sorted(all, key=lambda x: x[1], reverse=True)
                    for i, (query, metric_value, change) in enumerate(all):
                        change = change if cmd == 'rag' else change[len("refiner."):]
                        agg_all.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')
                        if metric_value > 0 and metric_value >= row[f'original.{ranker}.{metric}']: agg_gold.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')
                        if metric_value > 0 and metric_value > row[f'original.{ranker}.{metric}']: agg_plat.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')
                        if ('bt' in change) and metric_value < row[f'original.{ranker}.{metric}']: neg_exp.write(f'{row.qid}\t{change}\t{query}\t{metric_value}\n')

    @classmethod
    def build(cls, df, refiners, ranker, metric, output):
        base_model_name = 'original'
        ds_df = df.iloc[:, :3]  # the original query info
        ds_df['star_model_count'] = 0
        for idx, row in df.iterrows():
            star_models = dict()
            for model_name in refiners:
                if model_name == base_model_name: continue
                flag, sum = True, 0
                v = df.loc[idx, f'{model_name}.{ranker}.{metric}']
                v = v if not pd.isna(v) else 0
                v0 = df.loc[idx, f'{base_model_name}.{ranker}.{metric}']
                v0 = v0 if not pd.isna(v0) else 0
                if v <= v0: flag = False; continue
                sum += v ** 2
                if flag: star_models[model_name] = sum

            if len(star_models) > 0:
                ds_df.loc[idx, 'star_model_count'] = len(star_models.keys())
                star_models_sorted = {k: v for k, v in sorted(star_models.items(), key=lambda item: item[1], reverse=True)}
                for i, star_model in enumerate(star_models_sorted.keys()):
                    ds_df.loc[idx, f'method.{i + 1}'] = star_model.replace('refiner.', '')
                    ds_df.loc[idx, f'metric.{i + 1}'] = math.sqrt(star_models[star_model])
                    ds_df.loc[idx, f'query.{i + 1}'] = df.loc[idx, f'{star_model}']
            else:
                ds_df.loc[idx, 'star_model_count'] = 0
        ds_df.to_csv(output, index=False, encoding='UTF-8')

    @classmethod
    def box(cls, input, qrels, output, checks, sel_ref):
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
            df.to_csv(f'{output}/{c}.{sel_ref}.tsv', sep='\t', encoding='utf-8', index=False, header=False)
            print(f'{c}  has {df.shape[0]} queries\n')
            df = df.merge(qrels, on='qid', how='inner')
            df.to_csv(f'{output}/{c}.{sel_ref}.qrels.tsv', sep='\t', encoding='utf-8', index=False, header=False, columns=list(qrels.columns))


def hex_to_ansi(hex_color_code="", reset=False):
    if reset: return "\033[0m"
    else:
        hex_color_code = hex_color_code.lstrip('#')
        red = int(hex_color_code[0:2], 16)
        green = int(hex_color_code[2:4], 16)
        blue = int(hex_color_code[4:6], 16)
        return f'\033[38;2;{red};{green};{blue}m'
