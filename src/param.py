import random, os, numpy as np, multiprocessing
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['search', 'eval'], # steps of pipeline, ['finetune', 'predict', 'search', 'eval']
    't5model': 'base.gc',# 'small.local'
    'ranker': 'bm25',#'qld'
    'metric': 'ndcg', # 'map'
    'treclib': '"./trec_eval.9.0.4/trec_eval.exe"',#in non-windows, remove .exe, also for pytrec_eval, 'pytrec'
    'concat': False,#if more than one relevant doc, concat them all into one
    'msmarco.passage': {
        'index': '../data/raw/msmarco.passage/lucene-index.msmarco-v1-passage.20220131.9ea315/',
        'pairing': [None, 'doc', 'query']# [context={msmarco.passage.passage does not have userinfo}, input={query, doc, docs}, output={query, doc, docs}]
    },
    'aol': {
        'pairing': ['userid', 'query', 'doc']
    }
}