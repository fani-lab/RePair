import random, os, numpy as np, platform
import torch

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
extension = '.exe' if platform.system() == 'Windows' else ""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

settings = {
    'cmd': ['rag', 'search'],  # steps of pipeline, ['query_refinement', 'similarity', 'rag', 'search', 'rag_fusion', 'eval', 'agg', 'build', 'box','dense_retrieve', 'stats]
    'datalist': ['./../data/raw/robust04', './../data/raw/gov2', './../data/raw/antique', './../data/raw/dbpedia', './../data/raw/clueweb09b'], # ['./../data/raw/robust04', './../data/raw/gov2', './../data/raw/antique', './../data/raw/dbpedia', './../data/raw/clueweb09b']
    'domainlist': ['robust04', 'gov2', 'antique', 'dbpedia', 'clueweb09b'], # ['robust04', 'gov2', 'antique', 'dbpedia', 'clueweb09b']
    'fusion_category': ['global', 'local'],  # ['all', 'global', 'local', 'bt_nllb', 'bt']
    'fusion_method': 'rrf_multi_k',  # 'rrf_multi_k, rrf', 'condorcet', 'random'
    'ncore': 2,
    'ranker': ['bm25'],           # ['qld', 'bm25', 'tct_colbert']
    'metric': ['map'],  # any valid trec_eval.9.0.4 metrics like ['map', 'ndcg', 'recip_rank']
    'batch': None,              # search per batch of queries for IR search using pyserini, if None, search per query
    'topk': 100,                # number of retrieved documents for a query
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
        'qrels': '../data/raw/msmarco.passage/qrels.train.tsv',
        'dense_index': 'msmarco-passage-tct_colbert-hnsw',
        'extcorpus': 'orcas',
        'pairing': [None, 'docs', 'query'],  # [context={msmarco does not have userinfo}, input={query, doc, doc(s)}, output={query, doc, doc(s)}], s means concat of docs
        'lseq': {"inputs": 32, "targets": 256},  # query length and doc length for t5 model,
    },
    'aol-ia': {
        'index_item': ['title'],    # ['url'], ['title', 'url'], ['title', 'url', 'text']
        'index': '../data/raw/aol-ia/lucene-index/title/',  # change based on index_item
        'dense_index': '../data/raw/aol-ia/dense-index/tct_colbert.title/',  # change based on index_item
        'qrels': '../data/raw/aol-ia/qrels.train.tsv',
        'dense_encoder':'../data/raw/aol-ia/dense-encoder/tct_colbert.title/',  # change based on index_item
        'pairing': [None, 'docs', 'query'],  # [context={2 scenarios, one with userID and one without userID). input={'userid','query','doc(s)'} output={'query','doc(s)'}
        'extcorpus': 'msmarco.passage',
        'lseq': {"inputs": 32, "targets": 256},  # query length and doc length for t5 model,
        'filter': {'minql': 1, 'mindocl': 10}  # [min query length, min doc length], after merge queries with relevant 'index_item', if |query| <= minql drop the row, if |'index_item'| < mindocl, drop row
    },
    'robust04': {
        'index': '../data/raw/robust04/lucene-index.robust04.pos+docvectors+rawdocs',
        'dense_index': '../data/raw/robust04/faiss_index_robust04',
        'encoded': '../data/raw/robust04/encoded_robust04',
        'size': 528155,
        'topics': '../data/raw/robust04/topics.robust04.txt',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields
        'w_a': 1,  # OnFields
        'tokens': 148000000,
        'qrels': '../data/raw/robust04/qrels.robust04.txt',
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'gov2': {
        'index': '../data/raw/gov2/lucene-index.gov2.pos+docvectors+rawdocs',
        'size': 25000000,
        'topics': '../data/raw/gov2/topics.terabyte0{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'trec': ['terabyte04.701-750', 'terabyte05.751-800', 'terabyte06.801-850'],
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 4,  # OnFields
        'w_a': 0.25,  # OnFields
        'tokens': 17000000000,
        'qrels': '../data/raw/gov2/qrels.terabyte0{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'robust04',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'clueweb09b': {
        'index': '../data/raw/clueweb09b/lucene-index.cw09b.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../data/raw/clueweb09b/topics.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'trec': ['web.1-50', 'web.51-100', 'web.101-150', 'web.151-200'],
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 1,  # OnFields
        'w_a': 0,  # OnFields
        'tokens': 31000000000,
        'qrels': '../data/raw/clueweb09b/qrels.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'clueweb12b13': {
        'index': '../data/raw/clueweb12b13/lucene-index.cw12b13.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../data/raw/clueweb12b13/topics.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'trec': ['201-250', '251-300'],
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 4,  # OnFields
        'w_a': 0,  # OnFields
        'tokens': 31000000000,
        'qrels': '../data/raw/clueweb12b13/qrels.web.{}.txt',  # {} is a placeholder for subtopics in main.py -> run()
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'antique': {
        'index': '../data/raw/antique/lucene-index-antique',
        'size': 403000,
        'topics': '../data/raw/antique/topics.antique.txt',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 16000000,
        'qrels': '../data/raw/antique/qrels.antique.txt',
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'trec09mq': {
        'index': '/data/raw/clueweb09b/lucene-index.cw09b.pos+docvectors+rawdocs',
        'size': 50000000,
        'topics': '../ds/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 16000000,
        'qrels': '../data/raw/trec09mq/prels.20001-60000.prep',
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'dbpedia': {
        'index': '../data/raw/dbpedia/lucene-index-dbpedia-collection',
        'size': 4632359,
        'topics': '../data/raw/dbpedia/topics.dbpedia.txt',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 1,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 200000000,
        'qrels': '../data/raw/dbpedia/qrels.dbpedia.txt',
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
    'orcas': {
        'index': '../data/raw/orcas/lucene-index.msmarco-v1-doc.20220131.9ea315',
        'size': 50000000,
        # 'topics': '../data/raw/trec2009mq/prep/09.mq.topics.20001-60000.prep.tsv',
        'topics': '../data/raw/orcas/preprocess/orcas-I-2M_topics.prep',
        'prels': '',  # this will be generated after a retrieval {bm25, qld}
        'w_t': 2.25,  # OnFields # to be tuned
        'w_a': 1,  # OnFields # to be tuned
        'tokens': 16000000,
        'qrels': '../data/raw/orcas/preprocess/orcas-doctrain-qrels.prep',
        'extcorpus': 'gov2',  # AdaptOnFields
        'pairing': [None, None, None],
        'index_item': [],
    },
}

# Only for sparse indexing
anserini = {
    'path': './src/anserini/',
    'trec_eval': '../anserini/eval/trec_eval.9.0.4/trec_eval'
}

