import random, os, numpy as np, platform
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
extension = '.exe' if platform.system() == 'Windows' else ""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['stats'],   # steps of pipeline, ['pair', 'finetune', 'predict', 'search', 'eval','agg', 'box','dense_retrieve']
    'ncore': 2,
    't5model': 'base.gc',   # 'base.gc' on google cloud tpu, 'small.local' on local machine
    'iter': 5,                  # number of finetuning iteration for t5
    'nchanges': 5,              # number of changes to a query
    'ranker': 'bm25',           # 'qld', 'bm25', 'tct_colbert'
    'batch': None,               # search per batch of queries for IR search using pyserini, if None, search per query
    'topk': 100,                 # number of retrieved documents for a query
    'metric': 'recip_rank.10',             # any valid trec_eval.9.0.4 metric like map, ndcg, recip_rank, ...
    'large_ds': True,
    'treclib': f'"./trec_eval.9.0.4/trec_eval{extension}"',  # in non-windows, remove .exe, also for pytrec_eval, 'pytrec'
    'box': {'gold': 'refined_q_metric >= original_q_metric and refined_q_metric > 0',
            'platinum': 'refined_q_metric > original_q_metric',
            'diamond': 'refined_q_metric > original_q_metric and refined_q_metric == 1'},
    'msmarco.passage': {
        'index_item': ['passage'],
        'index': '../data/raw/msmarco.passage/lucene-index.msmarco-v1-passage.20220131.9ea315/',
        'dense_encoder': 'castorini/tct_colbert-msmarco',
        'dense_index': 'msmarco-passage-tct_colbert-hnsw',
        'pairing': [None, 'docs', 'query'],     # [context={msmarco does not have userinfo}, input={query, doc, doc(s)}, output={query, doc, doc(s)}], s means concat of docs
        'lseq': {"inputs": 32, "targets": 256},  # query length and doc length for t5 model,
    },
    'aol-ia': {
        'index_item': ['title'],    # ['url'], ['title', 'url'], ['title', 'url', 'text']
        'index': '../data/raw/aol-ia/lucene-index/title/',  # change based on index_item
        'dense_index': '../data/raw/aol-ia/dense-index/tct_colbert.title/',  # change based on index_item
        'dense_encoder':'../data/raw/aol-ia/dense-encoder/tct_colbert.title/',  # change based on index_item
        'pairing': [None, 'docs', 'query'],     # [context={2 scenarios, one with userID and one without userID). input={'userid','query','doc(s)'} output={'query','doc(s)'}
        'lseq': {"inputs": 32, "targets": 256},  # query length and doc length for t5 model,
        'filter': {'minql': 1, 'mindocl': 10}   # [min query length, min doc length], after merge queries with relevant 'index_item', if |query| <= minql drop the row, if |'index_item'| < mindocl, drop row
    }
}
