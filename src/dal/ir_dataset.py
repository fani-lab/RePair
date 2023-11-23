import json, os, sys, pandas as pd,numpy as np
from tqdm import tqdm
from shutil import copyfile
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from dal.ds import Dataset
from abc import abstractclassmethod
tqdm.pandas()

class IR_Dataset(Dataset):

    def __init__(self, settings, domain, homedir, ncore):
        try: super(IR_Dataset, self).__init__(settings=settings, domain=domain)
        except: self._build_index(homedir, settings, ncore)
    
    @classmethod
    def _build_index(cls, homedir, settings, ncore):
        indexdir = settings['index']
        index_item = settings['index_item']
        qrels_cols = settings['qrels_cols']
        queries_cols = settings['queries_cols']
        docs_cols = settings['docs_cols']
        dataset_name = settings['dataset_name']

        print("Creating index from scratch using ir-dataset...")
        os.environ['IR_DATASETS_HOME'] = '/'.join(homedir.split('/')[:-1])
        if not os.path.isdir(os.environ['IR_DATASETS_HOME']): os.makedirs(os.environ['IR_DATASETS_HOME'])
        index_item_str = '.'.join(index_item)
        if not os.path.isdir(f'{indexdir}/{cls.user_pairing}{index_item_str}'): os.makedirs(f'{indexdir}/{cls.user_pairing}{index_item_str}')
        ##if not os.path.isdir(f'{indexdir}/{index_item_str}'): os.makedirs(f'{indexdir}/{index_item_str}')
        import ir_datasets
        sys.path.append(os.path.dirname(os.getcwd()))
        from cmn import lucenex
        print(f"Setting up nfCorpus corpus using ir-datasets at {homedir}...")
        
        ## change this from test to train for the big dataset
        dataset = ir_datasets.load(dataset_name)
        print("Getting queries and qrels...")

        qrels = pd.DataFrame.from_records(dataset.qrels_iter(), columns=qrels_cols)
        queries = pd.DataFrame.from_records(dataset.queries_iter(), columns=queries_cols)
        docs = pd.DataFrame.from_records(dataset.docs_iter(), columns=docs_cols)

        qrels.to_csv(f'{homedir}/qrels.csv', index=False)
        queries.to_csv(f'{homedir}/queries.csv', index=False)
        docs.to_csv(f'{homedir}/docs.csv', index=False)

        print('Creating jsonl collections for indexing...')

        cls.create_jsonl(dataset, index_item, f'{homedir}/{index_item_str}')

        if len(os.listdir(f'{indexdir}/{cls.user_pairing}{index_item_str}')) == 0:
          ##lucenex(f'{homedir}/{index_item_str}', f'{indexdir}/{index_item_str}/', ncore)
          lucenex.lucenex(f'{homedir}/{cls.user_pairing}{index_item_str}', f'{indexdir}/{cls.user_pairing}{index_item_str}/', ncore)

        if os.path.isfile(f'{homedir}/qrels'): copyfile(f'{homedir}/qrels', f'{homedir}/qrels.test.tsv')
        if os.path.isfile(f'{homedir}/queries.tsv'): copyfile(f'{homedir}/queries.tsv', f'{homedir}/queries.test.tsv')
        cls.searcher = LuceneSearcher(f'{indexdir}/{index_item_str}')
        return

    @abstractclassmethod
    def read_queries(cls, input, domain):
        #This method is implemented by the subclasses that need it
        pass

    @abstractclassmethod
    def create_jsonl(cls, dataset, index_item, output):
        #This method is implemented by the subclasses that need it
        pass
    
    @abstractclassmethod
    def pair(cls, input, output, cat=True):
        #This method is implemented by the subclasses that need it.
        pass