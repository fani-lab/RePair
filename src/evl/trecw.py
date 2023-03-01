import os

def evaluate(in_docids, out_metrics, qrels, metric, lib, mean=False):
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
        # trec_eval.9.0.4 does not accept more than 2GB files!
        # So, we need to break it into several files.
        ranker = in_docids.split('.')[-1]
        size = os.path.getsize(in_docids)
        in_docids_list = [(in_docids, out_metrics)]
        if int(size / (2 * 2 ** 30)): #greater that 2GB
            from filesplit.split import Split
        # if size > 1:
            print(f'trec_eval does not accept more than 2GB files! Breaking input file {in_docids} into several splits ...')
            mean = False # since for each split we gonna have the result, it's challenging to merge the means
            splitdir = f'{in_docids}.splits/'
            if not os.path.isdir(splitdir): os.makedirs(splitdir)
            split = Split(in_docids, splitdir)
            split.splitdelimiter = '.'
            split.bylinecount(2*2**20)#each split 2MB lines of 1K char
            # split.bylinecount(100)
            in_docids_list = [(f'{splitdir}{d}', f'{splitdir}{d}.{metric}') for d in os.listdir(splitdir) if ranker in d]

        for i, o in in_docids_list:
            cli_cmd = f'{lib} {"-n" if not mean else ""} -q -m {metric} {qrels} {i} > {o}'
            print(cli_cmd)
            stream = os.popen(cli_cmd)
            print(stream.read())

        #merge the results for splitted files if any
        if len(in_docids_list) > 1:
            from filesplit.merge import Merge
            print('Merging the results of splits ...')
            m = Merge(splitdir, os.path.dirname(out_metrics), os.path.basename(out_metrics))
            m.merge(cleanup=True)
    else: raise NotImplementedError

# unit test
# evaluate('../../output/toy.msmarco.passage/t5.small.local.docs.query/original.bm25',
#          '../../output/toy.msmarco.passage/t5.small.local.docs.query/original.bm25.map',
#          '../../data/raw/toy.msmarco.passage/qrels.train.tsv_',
#          'map',
#          '../trec_eval.9.0.4/trec_eval',
#          mean=False)

