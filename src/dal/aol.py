import json, os, pandas as pd,numpy as np
from tqdm import tqdm
from shutil import copyfile
tqdm.pandas()

from pyserini.search.lucene import LuceneSearcher

from dal.ds import Dataset

class Aol(Dataset):

    def __init__(self, settings, homedir, ncore):
        try: super(Aol, self).__init__(settings=settings)
        except: self._build_index(homedir, Dataset.settings['index_item'], Dataset.settings['index'], ncore)

    @classmethod
    def _build_index(cls, homedir, index_item, indexdir, ncore):
        print(f"Creating index from scratch using ir-dataset ...")
        #https://github.com/allenai/ir_datasets
        os.environ['IR_DATASETS_HOME'] = '/'.join(homedir.split('/')[:-1])
        if not os.path.isdir(os.environ['IR_DATASETS_HOME']): os.makedirs(os.environ['IR_DATASETS_HOME'])
        if not os.path.isdir(indexdir): os.makedirs(indexdir)
        import ir_datasets
        from cmn.lucenex import lucenex

        print(f"Setting up aol corpus using ir-datasets at {homedir}...")
        aolia = ir_datasets.load("aol-ia")
        print('Getting queries and qrels ...')
        # the column order in the file is [qid, uid, did, uid]!!!! STUPID!!
        qrels = pd.DataFrame.from_records(aolia.qrels_iter(), columns=['qid', 'did', 'rel', 'uid'], nrows=1)  # namedtuple<query_id, doc_id, relevance, iteration>
        queries = pd.DataFrame.from_records(aolia.queries_iter(), columns=['qid', 'query'], nrows=1)# namedtuple<query_id, text>

        print('Creating jsonl collections for indexing ...')
        print(f'Raw documents should be downloaded already at {homedir}/aol-ia/downloaded_docs/ as explained here: https://github.com/terrierteam/aolia-tools')
        print(f'But it had bugs: https://github.com/allenai/ir_datasets/issues/222')
        print(f'Sean MacAvaney provided us with the downloaded_docs.tar file. Thanks Sean!')
        index_item_str = '.'.join(index_item)
        Aol.create_jsonl(aolia, index_item, f'{homedir}/{index_item_str}')
        if len(os.listdir(f'{indexdir}/{index_item_str}')) == 0:
            lucenex(f'{homedir}/{index_item_str}', f'{indexdir}/{index_item_str}', ncore)
        # do NOT rename qrel to qrel.tsv or anything else as aol-ia does not like it!!
        # if os.path.isfile(f'{homedir}/qrels'): os.rename(f'{homedir}/qrels', f'{homedir}/qrels')
        if os.path.isfile(f'{homedir}/qrels'): copyfile(f'{homedir}/qrels', f'{homedir}/qrels.train.tsv')
        if os.path.isfile(f'{homedir}/queries.tsv'): copyfile(f'{homedir}/queries.tsv', f'{homedir}/queries.train.tsv')
        cls.searcher = LuceneSearcher(f'{indexdir}/{index_item_str}')
        # dangerous cleaning!
        # for d in os.listdir(homedir):
        #     if not (d.find('aol-ia') > -1) and os.path.isdir(f'./../data/raw/{d}'): shutil.rmtree(f'./../data/raw/{d}')

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
                doc = {'title': doc.title, 'url': doc.url, 'text': doc.text}
                doc = ' '.join([doc[item] for item in index_item])
                output_jsonl_file.write(json.dumps({'id': did, 'contents': doc}) + '\n')
                if i % 100000 == 0: print(f'Converted {i:,} docs, writing into file {output_jsonl_file.name} ...')
            output_jsonl_file.close()

    @classmethod
    def pair(cls, input, output, cat=True):
        queries = pd.read_csv(f'{input}/queries.train.tsv', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
        # the column order in the file is [qid, uid, did, uid]!!!! STUPID!!
        qrels = pd.read_csv(f'{input}/qrels.train.tsv', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'uid', 'did', 'rel'], header=None)
        #not considering uid
        # docid is a hash of the URL. qid is the a hash of the *noramlised query* ==> two uid may have same qid then, same docid.
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)

        doccol = 'docs' if cat else 'doc'
        del queries
        queries_qrels['ctx'] = ''
        queries_qrels = queries_qrels.astype('category')
        queries_qrels[doccol] = queries_qrels['did'].progress_apply(cls._txt)

        # no uid for now + some cleansings ...
        queries_qrels.drop_duplicates(subset=['qid', 'did'], inplace=True)  # two users with same click for same query
        queries_qrels['uid'] = -1
        queries_qrels.dropna(inplace=True) #empty doctxt, query, ...
        queries_qrels.drop(queries_qrels[queries_qrels['query'].str.strip().str.len() <= Dataset.settings['filter']['minql']].index,inplace=True)
        queries_qrels.drop(queries_qrels[queries_qrels[doccol].str.strip().str.len() < Dataset.settings["filter"]['mindocl']].index,inplace=True)  # remove qrels whose docs are less than mindocl
        queries_qrels.to_csv(f'{input}/qrels.train.tsv_', index=False, sep='\t', header=False, columns=['qid', 'uid', 'did', 'rel'])

        if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False, observed=True).agg({'uid': list, 'did': list, doccol: ' '.join})
        queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
        batch_size = 1000000  # need to make this dynamic
        if len(queries_qrels) > batch_size:
            for _, chunk in queries_qrels.groupby(np.arange(queries_qrels.shape[0]) // batch_size):
                chunk.to_csv(f'../output/aol-ia/t5.base.gc.docs.query.title.url/original_test_queries/original.{_}.tsv',
                             sep='\t', encoding='utf-8', index=False, columns=['query'], header=False)
                qrels = chunk.merge(qrels, on='qid', how='inner')
                qrels.to_csv(f'../output/aol-ia/t5.base.gc.docs.query.title.url/qrels/qrels.splits.{_}.tsv_', sep='\t',
                             encoding='utf-8', index=False, header=False, columns=['qid', 'uid', 'did', 'rel'])
        return queries_qrels

