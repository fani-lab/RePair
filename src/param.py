import random, os, numpy as np
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['finetune', 'predict'], # steps of pipeline, ['finetune', 'predict', 'eval']
    'concat': False,#if more than one relevant doc, concat them all into one
    'msmarco-passage': {
        'pairing': [None, 'query', 'doc']# [context={msmarco does not have userinfo}, input={query, doc, docs}, output={query, doc, docs}]
    },
    'aol': {
        'pairing': ['userid', 'query', 'doc']
    }
}