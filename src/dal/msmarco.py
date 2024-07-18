import pandas as pd
from tqdm import tqdm
from dal.ds import Dataset
tqdm.pandas()


class MsMarcoPsg(Dataset):

    def __init__(self, settings, domain):
        super(MsMarcoPsg, self).__init__(settings=settings, domain=domain)
        MsMarcoPsg.user_pairing = "user/" if "user" in MsMarcoPsg.settings["pairing"] else ""
        MsMarcoPsg.index_item_str = '.'.join(MsMarcoPsg.settings["index_item"])


    @classmethod
    def read_queries(cls, input, domain, trec=False):
        queries = pd.read_csv(f'{input}/queries.train.tsv', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
        qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
        qrels.drop_duplicates(subset=['qid', 'pid'], inplace=True)  # qrels have duplicates!!
        qrels.to_csv(f'{input}/qrels.train.tsv_', index=False, sep='\t', header=False)  # trec_eval.9.0.4 does not accept duplicate rows!!
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
        queries_qrels = queries_qrels.sort_values(by='qid')
        cls.create_query_objects(queries_qrels, ['qid', 'did', 'pid', 'relevancy'], domain)

    @classmethod
    def set_index(cls, index): super().search_init(f'{index}{Dataset.user_pairing}{Dataset.index_item_str}')

    @classmethod
    def pair(cls, output, cat=True):
        # TODO: change the code in a way to use read_queries
        queries = pd.read_csv(f'{input}/queries.train.tsv', sep='\t', index_col=False, names=['qid', 'query'],converters={'query': str.lower}, header=None)
        qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
        qrels.drop_duplicates(subset=['qid', 'pid'], inplace=True)  # qrels have duplicates!!
        qrels.to_csv(f'{input}/qrels.train.tsv_', index=False, sep='\t', header=False)  # trec_eval.9.0.4 does not accept duplicate rows!!
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
        doccol = 'docs' if cat else 'doc'
        queries_qrels[doccol] = queries_qrels['pid'].apply(cls._txt)  # 100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
        queries_qrels['ctx'] = ''
        if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False, observed=True).agg({'did': list, 'pid': list, doccol: ' '.join})
        queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
        return queries_qrels
