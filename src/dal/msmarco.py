import os
from os.path import isfile,join
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from dal.ds import Dataset

class MsMarcoPsg(Dataset):

    def __init__(self, settings): super(MsMarcoPsg, self).__init__(settings=settings)

    @classmethod
    def pair(cls, input, output, cat=True):
        queries = pd.read_csv(f'{input}/queries.train.tsv', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
        qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
        qrels.drop_duplicates(subset=['qid', 'pid'], inplace=True)  # qrels have duplicates!!
        qrels.to_csv(f'{input}/qrels.train.tsv_', index=False, sep='\t', header=False) #trec_eval.9.0.4 does not accept duplicate rows!!
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
        doccol = 'docs' if cat else 'doc'
        queries_qrels[doccol] = queries_qrels['pid'].progress_apply(cls._txt) #100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
        queries_qrels['ctx'] = ''
        if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False, observed=True).agg({'did': list, 'pid': list, doccol: ' '.join})
        queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
        return queries_qrels

