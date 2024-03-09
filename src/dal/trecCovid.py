## "cord19/trec-covid" loader

import json, os, sys, pandas as pd,numpy as np
from tqdm import tqdm
from shutil import copyfile
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from dal.ir_dataset import IR_Dataset
tqdm.pandas()

class trecCovid(IR_Dataset):
    def __init__(self, settings, domain, homedir, ncore):
            try: super(trecCovid, self).__init__(settings=settings, domain=domain, homedir=homedir, ncore=ncore)
            except: self._build_index(homedir, IR_Dataset.settings, ncore)

    @classmethod
    def create_jsonl(cls, dataset, index_item, output):
        print(output)
        if not os.path.isdir(output): os.makedirs(output)
        if not os.path.isfile(f'{output}/docs.json'):
            print(f'Converting trecCovid docs into jsonl collection for {index_item}')
            output_jsonl_file = open(f'{output}/docs.json', 'w', encoding='utf-8', newline='\n')
            for i, doc in enumerate(dataset.docs_iter()):  # doc returns doc_id, title, text, url
                did = doc.doc_id
                #doc = {'title': fix_text(doc.title), 'url': doc.url, 'text': str(fix_text(doc.text))[:2000]}
                doc = {'title': fix_text(doc.title), 'doi': doc.doi, 'date': doc.date, 'abstract': doc.abstract}
                doc = ' '.join([doc[item] for item in index_item])
                output_jsonl_file.write(json.dumps({'id': did, 'contents': doc}) + '\n')
                if i % 100000 == 0: print(f'Converted {i:,} docs, writing into file {output_jsonl_file.name} ...')
            output_jsonl_file.close()
  
    @classmethod
    def read_queries(cls, input, domain):
        #This method is implemented by the subclasses that need it
        pass
    
    @classmethod
    def pair(cls, input, output, cat=True):
        #This method is not yet implemented for this class
        pass