##nfCorpus loader

import json, os, sys, pandas as pd,numpy as np
from tqdm import tqdm
from shutil import copyfile
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from dal.ds import Dataset
tqdm.pandas()

class nfCorpus(Dataset):
  def __init__(self, settings, homedir, ncore):
        try: super(nfCorpus, self).__init__(settings=settings)
        except: self._build_index(homedir, Dataset.settings['index_item'], Dataset.settings['index'], ncore)

  @classmethod
  def _build_index(cls, homedir, index_item, indexdir, ncore):
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
    nfCorpusSet = ir_datasets.load("beir/nfcorpus/test")
    print("Getting queries and qrels...")

    qrels = pd.DataFrame.from_records(nfCorpusSet.qrels_iter(), columns=['qid', 'did', 'rel', 'iteration'], nrows=1)
    queries = pd.DataFrame.from_records(nfCorpusSet.queries_iter(), columns=['qid', 'query', 'url'], nrows=1)

    print('Creating jsonl collections for indexing...')

    nfCorpus.create_jsonl(nfCorpusSet, index_item, f'{homedir}/{index_item_str}')

    if len(os.listdir(f'{indexdir}/{cls.user_pairing}{index_item_str}')) == 0:
      ##lucenex(f'{homedir}/{index_item_str}', f'{indexdir}/{index_item_str}/', ncore)
      lucenex.lucenex(f'{homedir}/{cls.user_pairing}{index_item_str}', f'{indexdir}/{cls.user_pairing}{index_item_str}/', ncore)

    if os.path.isfile(f'{homedir}/qrels'): copyfile(f'{homedir}/qrels', f'{homedir}/qrels.test.tsv')
    if os.path.isfile(f'{homedir}/queries.tsv'): copyfile(f'{homedir}/queries.tsv', f'{homedir}/queries.test.tsv')
    cls.searcher = LuceneSearcher(f'{indexdir}/{index_item_str}')
    return

  @classmethod
  def create_jsonl(cls, nfCorpusSet, index_item, output):
    print(output)
    if not os.path.isdir(output): os.makedirs(output)
    if not os.path.isfile(f'{output}/docs.json'):
        print(f'Converting nfCorpus docs into jsonl collection for {index_item}')
        print(output)
        output_jsonl_file = open(f'{output}/docs.json', 'w', encoding='utf-8', newline='\n')
        for i, doc in enumerate(nfCorpusSet.docs_iter()):  # doc returns doc_id, title, text, url
            did = doc.doc_id
            doc = {'title': fix_text(doc.title), 'url': doc.url, 'text': str(fix_text(doc.text))[:2000]}
            doc = ' '.join([doc[item] for item in index_item])
            output_jsonl_file.write(json.dumps({'id': did, 'contents': doc}) + '\n')
            if i % 100000 == 0: print(f'Converted {i:,} docs, writing into file {output_jsonl_file.name} ...')
        output_jsonl_file.close()