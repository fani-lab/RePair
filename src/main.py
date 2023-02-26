import argparse, os, pandas as pd, multiprocessing as mp
from functools import partial
from multiprocessing import freeze_support
from os import listdir
from os.path import isfile, join
from shutil import copyfile

import param

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    for domain in domain_list:
        datapath = data_list[domain_list.index(domain)]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_output): os.makedirs(prep_output)

        if domain == 'msmarco.passage':
            from dal.msmarco import MsMarcoPsg
            ds = MsMarcoPsg(param.settings[domain])
        if domain == 'aol-ia':
            from dal.aol import Aol
            ds = Aol(param.settings[domain], datapath, param.settings['ncore'])
        if domain == 'yandex' in domain_list: raise ValueError('Yandex is yet to be added ...')

        index_item_str = '.'.join(settings[domain]['index_item'])
        in_type, out_type = settings[domain]['pairing'][1], settings[domain]['pairing'][2]
        tsv_path = {'train': f'{prep_output}/{in_type}.{out_type}.{index_item_str}.train.tsv', 'test': f'{prep_output}/{in_type}.{out_type}.{index_item_str}.test.tsv'}

        query_qrel_doc = None
        if 'pair' in settings['cmd']:
            print(f'Pairing queries and relevant passages for training set ...')
            cat = True if 'docs' in {in_type, out_type} else False
            query_qrel_doc = ds.pair(datapath, f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.{index_item_str}.train.tsv', cat=cat)
            print(f'Pairing queries and relevant passages for test set ...')
            #TODO: query_qrel_doc = pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.{index_item_str}.test.tsv')
            query_qrel_doc = ds.pair(datapath, f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.{index_item_str}.test.tsv', cat=cat)
            query_qrel_doc.to_csv(tsv_path['train'], sep='\t', encoding='utf-8', index=False, columns=[in_type, out_type], header=False)
            query_qrel_doc.to_csv(tsv_path['test'], sep='\t', encoding='utf-8', index=False, columns=[in_type, out_type], header=False)

        t5_model = settings['t5model']  # {"small", "base", "large", "3B", "11B"} cross {"local", "gc"}
        t5_output = f'../output/{os.path.split(datapath)[-1]}/t5.{t5_model}.{in_type}.{out_type}.{index_item_str}'
        if not os.path.isdir(t5_output): os.makedirs(t5_output)
        copyfile('./param.py', f'{t5_output}/param.py')
        if {'finetune', 'predict'} & set(settings['cmd']):
            from mdl import mt5w
            if 'finetune' in settings['cmd']:
                print(f"Finetuning {t5_model} for {settings['iter']} iterations and storing the checkpoints at {t5_output} ...")
                mt5w.finetune(
                    tsv_path=tsv_path,
                    pretrained_dir=f'./../output/t5-data/pretrained_models/{t5_model.split(".")[0]}', #"gs://t5-data/pretrained_models/{"small", "base", "large", "3B", "11B"}
                    steps=settings['iter'],
                    output=t5_output, task_name=f"{domain.replace('-', '')}_cf",#:DD Task name must match regex: ^[\w\d\.\:_]+$
                    lseq=settings[domain]['lseq'],
                    nexamples=None, in_type=in_type, out_type=out_type, gcloud=False)

            if 'predict' in settings['cmd']:
                print(f"Predicting {settings['nchanges']} query changes using {t5_model} and storing the results at {t5_output} ...")
                mt5w.predict(
                    iter=settings['nchanges'],
                    split='test',
                    tsv_path=tsv_path,
                    output=t5_output,
                    lseq=settings[domain]['lseq'],
                    gcloud=False)

        if 'search' in settings['cmd']:
            print(f"Searching documents for query changes using {settings['ranker']} ...")
            #seems for some queries there is no qrels, so they are missed for t5 prediction.
            #query_originals = pd.read_csv(f'{datapath}/queries.train.tsv', sep='\t', names=['qid', 'query'], dtype={'qid': str})

            #we use the file after panda.merge that create the training set so we make sure the mapping of qids
            query_originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if "docs" in {in_type, out_type} else ""}.ctx.{index_item_str}.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            query_changes = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["ranker"]}') for f in listdir(t5_output) if isfile(join(t5_output, f)) and f.startswith('pred.') and len(f.split('.')) == 2]
            # for (i, o) in query_changes: ds.search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
            # batch search:
            # for (i, o) in query_changes: ds.search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
            # seems the LuceneSearcher cannot be shared in multiple processes! See dal.ds.py
            # parallel on each file ==> Problem: starmap does not understand inherited Dataset.searcher attribute!
            with mp.Pool(settings['ncore']) as p:
                p.starmap(partial(ds.search, qids=query_originals['qid'].values.tolist(), ranker=settings['ranker'], topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'], index=ds.searcher.index_dir), query_changes)

            # we need to add the original queries as well
            if not isfile(join(t5_output, f'original.{settings["ranker"]}')):
                query_originals.to_csv(f'{t5_output}/original', columns=['query'], index=False, header=False)
                ds.search_df(pd.DataFrame(query_originals['query']), f'{t5_output}/original.{settings["ranker"]}', query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'])

        if 'eval' in settings['cmd']:
            from evl import trecw
            search_results = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["metric"]}') for f in listdir(t5_output) if f.endswith(settings['ranker'])]

            if not isfile(f'{datapath}/qrels.train.tsv_'):
                qrels = pd.read_csv(f'{datapath}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
                qrels.drop_duplicates(subset=['qid', 'pid'], inplace=True)  # qrels have duplicates!!
                qrels.to_csv(f'{datapath}/qrels.train.tsv_', index=False, sep='\t', header=False)  # trec_eval.9.0.4 does not accept duplicate rows!!
            # for (i, o) in search_results: trecw.evaluate(i, o, qrels=f'{datapath}/qrels.train.tsv_', metric=settings['metric'], lib=settings['treclib'])
            with mp.Pool(settings['ncore']) as p:
                p.starmap(partial(trecw.evaluate, qrels=f'{datapath}/qrels.train.tsv_', metric=settings['metric'], lib=settings['treclib']), search_results)

        if 'agg' in settings['cmd']:
            originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if "docs" in {in_type, out_type} else ""}.ctx.{index_item_str}.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            original_metric_values = pd.read_csv(join(t5_output, f'original.{settings["ranker"]}.{settings["metric"]}'), sep='\t', usecols=[1,2], names=['qid', f'original.{settings["ranker"]}.{settings["metric"]}'], index_col=False, dtype={'qid': str})
            originals = originals.merge(original_metric_values, how='left', on='qid')
            originals[f'original.{settings["ranker"]}.{settings["metric"]}'].fillna(0, inplace=True)
            changes = [('.'.join(f.split('.')[0:2]), f) for f in os.listdir(t5_output) if f.endswith(f'{settings["ranker"]}.{settings["metric"]}') and 'original' not in f]
            ds.aggregate(originals, changes, t5_output)

        if 'box' in settings['cmd']:
            box_path = join(t5_output, f'{settings["ranker"]}.{settings["metric"]}.boxes')
            if not os.path.isdir(box_path): os.makedirs(box_path)
            gold_df = pd.read_csv(f'{t5_output}/{settings["ranker"]}.{settings["metric"]}.agg.all.tsv', sep='\t', header=0, dtype={'qid': str})
            qrels = pd.read_csv(f'{datapath}/qrels.train.tsv_', names=['qid', 'did', 'pid', 'rel'], sep='\t', dtype={'qid': str})

            box_condition = settings['box']
            ds.box(gold_df, qrels, box_path, box_condition)
            for c in box_condition.keys():
                print(f'Stamping boxes for {settings["ranker"]}.{settings["metric"]} before and after refinements ...')
                from evl import trecw
                if not os.path.isdir(join(box_path, 'stamps')): os.makedirs(join(box_path, 'stamps'))

                df = pd.read_csv(f'{box_path}/{c}.tsv', sep='\t', encoding='utf-8', index_col=False, header=None, names=['qid', 'query', 'metric', 'query_', 'metric_'], dtype={'qid': str})
                df.drop_duplicates(subset=['qid'], inplace=True)#See ds.boxing(): in case we store more than two changes with the same metric value
                ds.search_df(df['query'].to_frame(), f'{box_path}/stamps/{c}.original.{settings["ranker"]}', df['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'])
                trecw.evaluate(f'{box_path}/stamps/{c}.original.{settings["ranker"]}', f'{box_path}/stamps/{c}.original.{settings["ranker"]}.{settings["metric"]}', qrels=f'{datapath}/qrels.train.tsv_', metric=settings['metric'], lib=settings['treclib'], mean=True)
                ds.search_df(df['query_'].to_frame().rename(columns={'query_': 'query'}), f'{box_path}/stamps/{c}.change.{settings["ranker"]}', df['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'])
                trecw.evaluate(f'{box_path}/stamps/{c}.change.{settings["ranker"]}', f'{box_path}/stamps/{c}.change.{settings["ranker"]}.{settings["metric"]}', qrels=f'{datapath}/qrels.train.tsv_', metric=settings['metric'], lib=settings['treclib'], mean=True)

def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -data ./../data/raw/toy.msmarco.passage)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -domain msmarco.passage)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='../output/', help='The output path (default: -output ../output/)')

# python -u main.py -data ../data/raw/toy.msmarco.passage -domain msmarco.passage
# python -u main.py -data ../data/raw/toy.aol-ia -domain aol-ia
# python -u main.py -data ../data/raw/toy.msmarco.passage ../data/raw/toy.aol-ia -domain msmarco.passage aol-ia

if __name__ == '__main__':
    freeze_support()
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Personalized Query Refinement')
    addargs(parser)
    args = parser.parse_args()

    # run(data_list=args.data_list,
    #         domain_list=args.domain_list,
    #         output=args.output,
    #         settings=param.settings)

    #after finetuning and predict, we can benchmark on rankers and metrics
    from itertools import product
    for ranker, metric in product(['bm25', 'qld'], ['success.10', 'map', 'recip_rank.10']):
        param.settings['ranker'] = ranker
        param.settings['metric'] = metric
        run(data_list=args.data_list,
            domain_list=args.domain_list,
            output=args.output,
            settings=param.settings)
