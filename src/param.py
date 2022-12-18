import random, os, numpy as np, multiprocessing
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['search', 'eval'], # steps of pipeline, ['finetune', 'predict', 'search', 'eval']
    'ranker': 'bm25',#'qld'
    'metric': {'success', 'ndcg_cut', 'map_cut'},
    'concat': False,#if more than one relevant doc, concat them all into one
    'msmarco.passage': {
        'index': '../data/raw/msmarco.passage/lucene-index.msmarco-v1-passage.20220131.9ea315/',
        'pairing': [None, 'query', 'doc']# [context={msmarco.passage.passage does not have userinfo}, input={query, doc, docs}, output={query, doc, docs}]
    },
    'aol': {
        'pairing': ['userid', 'query', 'doc']
    }
}