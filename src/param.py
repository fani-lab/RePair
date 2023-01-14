import random, os, numpy as np, multiprocessing
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['aggregate'],# steps of pipeline, ['finetune', 'predict', 'search', 'eval','aggregate']
    't5model': 'base.gc',#'base.gc',# 'small.local'
    'ranker': 'bm25',#'qld'
    'metric': 'map',# 'map'
    'treclib': '"./trec_eval.9.0.4/trec_eval.exe"',#in non-windows, remove .exe, also for pytrec_eval, 'pytrec'
    'msmarco.passage': {
        'index': '../data/raw/msmarco.passage/lucene-index.msmarco-v1-passage.20220131.9ea315/',
        'pairing': [None, 'docs', 'query']# [context={msmarco does not have userinfo}, input={query, doc, doc(s)}, output={query, doc, doc(s)}], s means concat of docs
    },
    'aol': {
        'index_item': 'title',# acceptable values 'text','title'
        'index': '../data/raw/aol/indexes/',
        'pairing': ['userid', 'query', 'doc']
    }
}