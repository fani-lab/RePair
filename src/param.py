import random, os, numpy as np, platform
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
extension = '.exe' if platform.system() == 'Windows' else ""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'query_refinement': True,
    'cmd': ['pair', 'finetune', 'predict', 'search', 'eval','agg', 'box'],   # steps of pipeline, ['pair', 'finetune', 'predict', 'search', 'eval','agg', 'box','dense_retrieve', 'stats]
    'ncore': 2,
    't5model': 'small.local',   # 'base.gc' on google cloud tpu, 'small.local' on local machine
    'iter': 5,                  # number of finetuning iteration for t5
    'nchanges': 5,              # number of changes to a query
    'ranker': 'bm25',           # 'qld', 'bm25', 'tct_colbert'
    'batch': None,               # search per batch of queries for IR search using pyserini, if None, search per query
    'topk': 100,                 # number of retrieved documents for a query
    'metric': 'map',             # any valid trec_eval.9.0.4 metric like map, ndcg, recip_rank, ...
    'large_ds': False,
    'treclib': f'"./trec_eval.9.0.4/trec_eval{extension}"',  # in non-windows, remove .exe, also for pytrec_eval, 'pytrec'
    'box': {'gold': 'refined_q_metric >= original_q_metric and refined_q_metric > 0',
            'platinum': 'refined_q_metric > original_q_metric',
            'diamond': 'refined_q_metric > original_q_metric and refined_q_metric == 1'}
}

corpora = {
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
    },
    'robust04': {
        'index': '../ds/robust04/lucene-index.robust04.pos+docvectors+rawdocs',
        'dense_index': '../ds/robust04/faiss_index_robust04',
        'encoded': '../ds/robust04/encoded_robust04',
        'size': 528155,
        'topics': '../ds/robust04/topics.robust04.txt',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields
        'w_a': 1,  # OnFields
        'tokens': 148000000,
        'qrels': '../ds/robust04/qrels.robust04.txt',
        'extcorpus': 'gov2',  # AdaptOnFields
    },
    'gov2': {
        'index': '../ds/gov2/lucene-index.gov2.pos+docvectors+rawdocs',
        'size': 25000000,
        'topics': '../ds/gov2/topics.terabyte0{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 4,  # OnFields
        'w_a': 0.25,  # OnFields
        'tokens': 17000000000,
        'qrels': '../ds/gov2/qrels.terabyte0{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'robust04',  # AdaptOnFields
    },
    'clueweb09b': {
        'index': '../ds/clueweb09b/lucene-index.cw09b.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../ds/clueweb09b/topics.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 1,  # OnFields
        'w_a': 0,  # OnFields
        'tokens': 31000000000,
        'qrels': '../ds/clueweb09b/qrels.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'gov2',  # AdaptOnFields
    },
    'clueweb12b13': {
        'index': '../ds/clueweb12b13/lucene-index.cw12b13.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../ds/clueweb12b13/topics.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 4,  # OnFields
        'w_a': 0,  # OnFields
        'tokens': 31000000000,
        'qrels': '../ds/clueweb12b13/qrels.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'gov2',  # AdaptOnFields
    },
    'antique': {
        'index': '../ds/antique/lucene-index-antique',
        'size': 403000,
        'topics': '../ds/antique/topics.antique.txt',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 16000000,
        'qrels': '../ds/antique/qrels.antique.txt',
        'extcorpus': 'gov2',  # AdaptOnFields
    },
    'trec09mq': {
        'index': 'D:\clueweb09b\lucene-index.cw09b.pos+docvectors+rawdocs',
        'size': 50000000,
        # 'topics': '../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv',
        'topics': '../ds/trec09mq/09.mq.topics.20001-60000.prep',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 16000000,
        'qrels': '../ds/trec09mq/prels.20001-60000.prep',
        'extcorpus': 'gov2',  # AdaptOnFields
    },
    'dbpedia': {
        'index': '../ds/dbpedia/lucene-index-dbpedia-collection',
        'size': 4632359,
        'topics': '../ds/dbpedia/topics.dbpedia.txt',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 1,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 200000000,
        'qrels': '../ds/dbpedia/qrels.dbpedia.txt',
        'extcorpus': 'gov2',  # AdaptOnFields
    },
    'orcas': {
        'index': '../ds/orcas/lucene-index.msmarco-v1-doc.20220131.9ea315',
        'size': 50000000,
        # 'topics': '../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv',
        'topics': '../ds/orcas/preprocess/orcas-I-2M_topics.prep',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 16000000,
        'qrels': '../ds/orcas/preprocess/orcas-doctrain-qrels.prep',
        'extcorpus': 'gov2',  # AdaptOnFields
    },
}

# Only for sparse indexing
anserini = {
    'path': '../anserini/',
    'trec_eval': '../anserini/eval/trec_eval.9.0.4/trec_eval'
}

