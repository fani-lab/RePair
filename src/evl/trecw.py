import os

def evaluate(in_docids, out_metrics, qrels, metric, lib, mean=False, topk=10):
    #qrels can have queries that are not in in_docids (superset)
    #also prediction may have queries that are not known to qrels
    # with open('pred', 'w') as f:
    #   f.write(f'1\tQ0\t2\t1\t20.30781\tPyserini Batch\n')
    #   f.write(f'1\tQ0\t3\t1\t5.30781\tPyserini Batch\n')
    #   f.write(f'5\tQ0\t3\t1\t5.30781\tPyserini Batch\n')#does not exist in qrel
    # with open('qrel', 'w') as f:
    #   f.write(f'1\t0\t2\t1\n')
    #   f.write(f'3\t0\t3\t1\n')#does not exist in prediction
    #"./../trec_eval.9.0.4/trec_eval" -q -m ndcg qrel pred
    # ndcg                    1       1.0000
    # ndcg                    all     1.0000
    #
    #However, no duplicate [qid, docid] can be in qrels!!

    print(f'Evaluating retrieved docs for {in_docids} with {metric} ...')
    if 'trec_eval' in lib:
        cli_cmd = f'{lib} {"-n" if not mean else ""} -q -m {metric} {qrels} {i} > {o}'
        print(cli_cmd)
        stream = os.popen(cli_cmd)
        print(stream.read())
    else: raise NotImplementedError

# unit test
# evaluate('../../output/toy.msmarco.passage/t5.small.local.docs.query/original.bm25',
#          '../../output/toy.msmarco.passage/t5.small.local.docs.query/original.bm25.map',
#          '../../data/raw/toy.msmarco.passage/qrels.train.tsv_',
#          'map',
#          '../trec_eval.9.0.4/trec_eval',
#          mean=False)

