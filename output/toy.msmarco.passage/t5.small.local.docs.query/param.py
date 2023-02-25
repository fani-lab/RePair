import random, os, numpy as np, platform, multiprocessing
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
extension = '.exe' if platform.system() == 'Windows' else ""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['pair', 'search', 'eval','agg', 'box'],# steps of pipeline, ['pair', 'finetune', 'predict', 'search', 'eval','agg', 'box']
    'ncore': multiprocessing.cpu_count(),
    't5model': 'small.local',#'base.gc', 'small.local'
    'iter': 5,          #number of finetuning iteration for t5
    'nchanges': 5,      #number of changes to a query
    'ranker': 'bm25',   #'qld', 'bm25'
    'batch': None,      #search per batch of queries for IR search using pyserini, if None, search per query
    'topk': 10,         #number of retrieved documents for a query
    'metric': 'map',    # any valid trec_eval metric like map, ndcg, recip_rank, ...
    'treclib': f'"./trec_eval.9.0.4/trec_eval{extension}"',#in non-windows, remove .exe, also for pytrec_eval, 'pytrec'
    'msmarco.passage': {
        'index_item': ['passage'],
        'index': '../data/raw/msmarco.passage/lucene-index.msmarco-v1-passage.20220131.9ea315/',
        'pairing': [None, 'docs', 'query'],     #[context={msmarco does not have userinfo}, input={query, doc, doc(s)}, output={query, doc, doc(s)}], s means concat of docs
        'lseq':{"inputs": 32, "targets": 256},  #query length and doc length for t5 model,
    },
    'aol-ia': {
        'index_item': ['title'], # ['url'], ['title', 'url'], ['title', 'url', 'text']
        'index': f'../data/raw/aol-ia/lucene-index/title/',
        'pairing': [None, 'docs', 'query'], #[context={2 scenarios, one with userID and one without userID). input={'userid','query','doc(s)'} output={'query','doc(s)'}
        'lseq':{"inputs": 32, "targets": 256},  #query length and doc length for t5 model,
        'filter': {'minql': 1, 'mindocl': 10}# [min query length, min doc length], after merge queries with relevant 'index_item', if |query| <= minql drop the row, if |'index_item'| < mindocl, drop row
    }
}