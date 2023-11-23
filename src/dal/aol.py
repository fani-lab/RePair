import json, os, pandas as pd,numpy as np
from tqdm import tqdm
from shutil import copyfile
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from dal.ir_dataset import IR_Dataset
from dal.query import Query
tqdm.pandas()


class Aol(IR_Dataset):

    def __init__(self, settings, domain, homedir, ncore):
        try: super(Aol, self).__init__(settings=settings, homedir=homedir, domain=domain, ncore=ncore)
        except: self._build_index(homedir, IR_Dataset.settings, ncore)

    @classmethod
    def read_queries(cls, input, domain):
        queries = pd.read_csv(f'{input}/queries.train.tsv', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
        # the column order in the file is [qid, uid, did, uid]!!!! STUPID!!
        qrels = pd.read_csv(f'{input}/qrels.train.tsv', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'uid', 'did', 'rel'], header=None)
        qrels.to_csv(f'{input}/qrels.train.tsv_', index=False, sep='\t', header=False)
        # docid is a hash of the URL. qid is the a hash of the *noramlised query* ==> two uid may have same qid then, same docid.
        qrels.drop_duplicates(subset=['qid', 'did', 'uid'], inplace=True)
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
        queries_qrels = queries_qrels.sort_values(by='qid')
        cls.create_query_objects(queries_qrels, ['qid', 'uid', 'did', 'rel'], domain)

    @classmethod
    def create_jsonl(cls, aolia, index_item, output):
        """
        https://github.com/castorini/anserini-tools/blob/7b84f773225b5973b4533dfa0aa18653409a6146/scripts/msmarco/convert_collection_to_jsonl.py
        :param index_item: defaults to title_and_text, use the params to create specified index
        :param output: folder name to create docs
        :return: list of docids that have empty body based on the index_item
        """
        if not os.path.isdir(output): os.makedirs(output)
        if not os.path.isfile(f'{output}/docs.json'):
            print(f'Converting aol docs into jsonl collection for {index_item}')
            output_jsonl_file = open(f'{output}/docs.json', 'w', encoding='utf-8', newline='\n')
            for i, doc in enumerate(aolia.docs_iter()):  # doc returns doc_id, title, text, url, ia_url
                did = doc.doc_id
                doc = {'title': fix_text(doc.title), 'url': doc.url, 'text': str(fix_text(doc.text))[:2000]}
                doc = ' '.join([doc[item] for item in index_item])
                output_jsonl_file.write(json.dumps({'id': did, 'contents': doc}) + '\n')
                if i % 100000 == 0: print(f'Converted {i:,} docs, writing into file {output_jsonl_file.name} ...')
            output_jsonl_file.close()

    @classmethod
    def pair(cls, queries, output, cat=True):
        # TODO: change the code in a way to use read_queries
        queries = pd.read_csv(f'{input}/queries.train.tsv', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
        # the column order in the file is [qid, uid, did, uid]!!!! STUPID!!
        qrels = pd.read_csv(f'{input}/qrels.train.tsv', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'uid', 'did', 'rel'], header=None)
        # docid is a hash of the URL. qid is the a hash of the *noramlised query* ==> two uid may have same qid then, same docid.
        qrels.drop_duplicates(subset=['qid', 'did', 'uid'], inplace=True)
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)

        doccol = 'docs' if cat else 'doc'
        del queries
        # in the event of user oriented pairing, we simply concat qid and uid
        if cls.user_pairing: queries_qrels['qid'] = queries_qrels['qid'].astype(str) + "_" + queries_qrels['uid'].astype(str)
        queries_qrels = queries_qrels.astype('category')
        queries_qrels[doccol] = queries_qrels['did'].apply(cls._txt)

        # queries_qrels.drop_duplicates(subset=['qid', 'did','pid'], inplace=True)  # two users with same click for same query
        if not cls.user_pairing: queries_qrels['uid'] = -1
        queries_qrels['ctx'] = ''
        queries_qrels.dropna(inplace=True)  # empty doctxt, query, ...
        queries_qrels.drop(queries_qrels[queries_qrels['query'].str.strip().str.len() <= Dataset.settings['filter']['minql']].index, inplace=True)
        queries_qrels.drop(queries_qrels[queries_qrels[doccol].str.strip().str.len() < Dataset.settings["filter"]['mindocl']].index, inplace=True)  # remove qrels whose docs are less than mindocl
        queries_qrels.drop_duplicates(subset=['qid', 'did'], inplace=True)
        queries_qrels.to_csv(f'{input}/{cls.user_pairing}qrels.train.tsv_', index=False, sep='\t', header=False, columns=['qid', 'uid', 'did', 'rel'])

        if cat: queries_qrels = queries_qrels.groupby(['qid', 'query', 'uid'], as_index=False, observed=True).agg({'did': list, doccol: ' '.join})
        queries_qrels[doccol] = queries_qrels['uid'].astype(str) + ": " + queries_qrels[doccol].astype(str)
        queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
        # queries_qrels = pd.read_csv(output, sep='\t', encoding='utf-8', index_col=False)
        if cls.user_pairing: qrels = pd.read_csv(f'{input}/{cls.user_pairing}qrels.train.tsv_', sep='\t', index_col=False, names=['qid', 'uid', 'did', 'rel'])
        batch_size = 1000000  # need to make this dynamic
        index_item_str = '.'.join(cls.settings['index_item'])
        # create dirs:
        if not os.path.isdir(f'../output/aol-ia/{cls.user_pairing}t5.base.gc.docs.query.{index_item_str}/original_test_queries'): os.makedirs(f'../output/aol-ia/{cls.user_pairing}t5.base.gc.docs.query.{index_item_str}/original_test_queries')
        if not os.path.isdir(f'../output/aol-ia/{cls.user_pairing}t5.base.gc.docs.query.{index_item_str}/qrels'): os.makedirs(f'../output/aol-ia/{cls.user_pairing}t5.base.gc.docs.query.{index_item_str}/qrels')
        if len(queries_qrels) > batch_size:
            for _, chunk in queries_qrels.groupby(np.arange(queries_qrels.shape[0]) // batch_size):
                chunk.to_csv(f'../data/preprocessed/aol-ia/{cls.user_pairing}docs.query.{index_item_str}.{_}.tsv', columns=['docs', 'query'], header=False, sep='\t', encoding='utf-8', index=False)
                chunk.to_csv(f'../output/aol-ia/{cls.user_pairing}t5.base.gc.docs.query.{index_item_str}/original/original.{_}.tsv',
                             sep='\t', encoding='utf-8', index=False, columns=['query'], header=False)
                qrels_splits = chunk[['qid', 'query']].merge(qrels, on='qid', how='inner')
                qrels_splits.to_csv(f'../output/aol-ia/{cls.user_pairing}t5.base.gc.docs.query.{index_item_str}/qrels/qrels.splits.{_}.tsv_', sep='\t',
                             encoding='utf-8', index=False, header=False, columns=['qid', 'uid', 'did', 'rel'])
        return queries_qrels
