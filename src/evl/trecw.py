import os


def evaluate(in_docids, out_metrics, qrels, metric, lib='pytrec'):#or 'pytrec'
    #qrels can have queries that are not in in_docids (superset)
    #also prediction may have queries that are not known to qrels
    # with open('pred', 'w') as f:
    #   f.write(f'1\tQ0\t2\t1\t20.30781\tPyserini Batch\n')
    #   f.write(f'1\tQ0\t3\t1\t5.30781\tPyserini Batch\n')
    #   f.write(f'5\tQ0\t3\t1\t5.30781\tPyserini Batch\n')#does not exist in qrel
    # with open('qrel', 'w') as f:
    #   f.write(f'1\t0\t2\t1\n')
    #   f.write(f'3\t0\t3\t1\n')#does not exist in prediction
    #"./../trec_eval.9.0.4/trec_eval.exe" -q -m ndcg qrel pred
    # ndcg                    1       1.0000
    # ndcg                    all     1.0000

    print(f'Evaluating retrieved docs for {in_docids} ...')
    if lib == 'pytrec':
        raise NotImplementedError
    else:
        cli_cmd = f'{lib} -q -m {metric} {qrels} {in_docids} > {out_metrics}'
        print(cli_cmd)
        stream = os.popen(cli_cmd)
        print(stream.read())

