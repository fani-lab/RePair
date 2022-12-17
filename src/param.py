import random
import torch
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed(0)

settings = {
    'cmd': ['predict'], # steps of pipeline, ['finetune', 'predict', 'eval']
    'concat': False,#if more than one relevant doc, concat them all into one
    'msmarco-passage': {
        'pairing': [None, 'query', 'doc']# [None, 'doc', 'query'], # [context, input, output]
    },
}