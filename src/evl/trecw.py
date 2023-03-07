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

        #we need to break it into several files for usage in low memory computers
        ranker = in_docids.split('.')[-1]
        size = os.path.getsize(in_docids)
        in_docids_list = [(in_docids, out_metrics)]
        if int(size / (2 * 2 ** 30)): #greater that 2GB
            from filesplit.split import Split
            print(f'Breaking input file {in_docids} into several splits for usage in low memory computers ...')
            mean = False # since for each split we gonna have the result, it's challenging to merge the means
            splitdir = f'{in_docids}.splits/'
            if not os.path.isdir(splitdir): os.makedirs(splitdir)
            split = Split(in_docids, splitdir)
            split.splitdelimiter = '.'
            # split.bylinecount((2*2**20) - ((2*2**20) % topk))#each split 2MB lines of 1K char but as exact multiplier of topk
            # small bug: even if we make the split based on topk, there may be a query with less topk relevant doc.
            # todo: instead of using filespli lib, opening the file and creating the splits manually
            split.bylinecount(2 * 2 ** 20)
            # split.bylinecount(100)
            in_docids_list = [(f'{splitdir}{d}', f'{splitdir}{d}.{metric}') for d in os.listdir(splitdir) if len(d.split('.')) == 4]

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
            m.merge(cleanup=False)
    else: raise NotImplementedError

# unit test
# evaluate('../../output/toy.msmarco.passage/t5.small.local.docs.query/original.bm25',
#          '../../output/toy.msmarco.passage/t5.small.local.docs.query/original.bm25.map',
#          '../../data/raw/toy.msmarco.passage/qrels.train.tsv_',
#          'map',
#          '../trec_eval.9.0.4/trec_eval',
#          mean=False)

