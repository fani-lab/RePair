import argparse, os, pandas as pd, multiprocessing
from functools import partial
from multiprocessing import freeze_support
from os import listdir
from os.path import isfile, join
from shutil import copyfile

from cmn.create_index import create_index
import param

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    if 'msmarco.passage' in domain_list:

        from dal import msmarco
        # seems the LuceneSearcher cannot be shared in multiple processes! See dal.msmarco.py

        datapath = data_list[domain_list.index('msmarco.passage')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_output): os.makedirs(prep_output)
        in_type, out_type = settings['msmarco.passage']['pairing'][1], settings['msmarco.passage']['pairing'][2]
        tsv_path = {'train': f'{prep_output}/{in_type}.{out_type}.train.tsv', 'test': f'{prep_output}/{in_type}.{out_type}.test.tsv'}

        query_qrel_doc = None
        if 'pair' in settings['cmd']:
            print(f'Pairing queries and relevant passages for training set ...')
            cat = True if 'docs' in {in_type, out_type} else False
            query_qrel_doc = msmarco.to_pair(datapath, f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.train.tsv', cat=cat)
            print(f'Pairing queries and relevant passages for test set ...')
            #TODO: query_qrel_doc = to_pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.test.tsv')
            query_qrel_doc = msmarco.to_pair(datapath, f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.test.tsv', cat=cat)
            query_qrel_doc.to_csv(tsv_path['train'], sep='\t', encoding='utf-8', index=False, columns=[in_type, out_type], header=False)
            query_qrel_doc.to_csv(tsv_path['test'], sep='\t', encoding='utf-8', index=False, columns=[in_type, out_type], header=False)

        t5_model = settings['t5model']  # {"small", "base", "large", "3B", "11B"} cross {"local", "gc"}
        t5_output = f'../output/{os.path.split(datapath)[-1]}/t5.{t5_model}.{in_type}.{out_type}'
        copyfile('./param.py', f'{t5_output}/param.py')
        if {'finetune', 'predict'} & set(settings['cmd']):
            from mdl import mt5w
            if 'finetune' in settings['cmd']:
                print(f"Finetuning {t5_model} for {settings['iter']} and storing the checkpoints at {t5_output} ...")
                mt5w.finetune(
                    tsv_path=tsv_path,
                    pretrained_dir=f'./../output/t5-data/pretrained_models/{t5_model.split(".")[0]}', #"gs://t5-data/pretrained_models/{"small", "base", "large", "3B", "11B"}
                    steps=settings['iter'],
                    output=t5_output, task_name='msmarco_passage_cf',
                    lseq=settings['msmarco.passage']['lseq'],
                    nexamples=None, in_type=in_type, out_type=out_type, gcloud=False)

            if 'predict' in settings['cmd']:
                print(f"Predicting {settings['nchanges']} query changes using {t5_model} and storing the results at {t5_output} ...")
                mt5w.predict(
                    iter=settings['nchanges'],
                    split='test',
                    tsv_path=tsv_path,
                    output=t5_output,
                    lseq=settings['msmarco.passage']['lseq'],
                    gcloud=False)

        if 'search' in settings['cmd']:
            print(f"Searching documents for query changes using {settings['ranker']} ...")

            #seems for some queries there is no qrels, so they are missed for t5 prediction.
            #query_originals = pd.read_csv(f'{datapath}/queries.train.tsv', sep='\t', names=['qid', 'query'], dtype={'qid': str})

            #we use the file after panda.merge that create the training set so we make sure the mapping of qids
            query_originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if "docs" in {in_type, out_type} else ""}.ctx.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            query_changes = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["ranker"]}') for f in listdir(t5_output) if isfile(join(t5_output, f)) and f.startswith('pred.') and len(f.split('.')) == 2]
            # for (i, o) in query_changes: msmarco.to_search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
            # batch search: for (i, o) in query_changes: msmarco.to_search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
            # parallel on each file
            with multiprocessing.Pool(settings['ncore']) as p:
                p.starmap(partial(msmarco.to_search, qids=query_originals['qid'].values.tolist(), ranker=settings['ranker'], topk=settings['topk'], batch=settings['batch']), query_changes)

            # we need to add the original queries as well
            if not isfile(join(t5_output, f'original.{settings["ranker"]}')):
                query_originals.to_csv(f'{t5_output}/original', columns=['query'], index=False, header=False)
                msmarco.to_search_df(pd.DataFrame(query_originals['query']), f'{t5_output}/original.{settings["ranker"]}', query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])

        if 'eval' in settings['cmd']:
            from evl import trecw
            search_results = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["metric"]}') for f in listdir(t5_output) if f.endswith(settings['ranker'])]

            if not isfile(f'{datapath}/qrels.train.nodups.tsv'):
                qrels = pd.read_csv(f'{datapath}/qrels.train.tsv', sep='\t', index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
                qrels.drop_duplicates(subset=['qid', 'pid'], inplace=True)  # qrels have duplicates!!
                qrels.to_csv(f'{datapath}/qrels.train.nodups.tsv', index=False, sep='\t', header=False)  # trec_eval does not accept duplicate rows!!
            # for (i, o) in search_results: trecw.evaluate(i, o, qrels=f'{datapath}/qrels.train.nodups.tsv', metric=settings['metric'], lib=settings['treclib'])
            with multiprocessing.Pool(settings['ncore']) as p:
                p.starmap(partial(trecw.evaluate, qrels=f'{datapath}/qrels.train.nodups.tsv', metric=settings['metric'], lib=settings['treclib']), search_results)

        if 'agg' in settings['cmd']:
            query_originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if "docs" in {in_type, out_type} else ""}.ctx.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': int})
            original_metric_values = pd.read_csv(join(t5_output, f'original.{settings["ranker"]}.{settings["metric"]}'), sep='\t', usecols=[1,2], names=['qid', f'original.{settings["ranker"]}.{settings["metric"]}'], index_col=False, skipfooter=1)
            original_metric_values[f'original.{settings["ranker"]}.{settings["metric"]}'].fillna(0, inplace=True)
            query_originals = query_originals.merge(original_metric_values, how='left', on='qid')
            query_changes = [('.'.join(f.split('.')[0:2]), f) for f in os.listdir(t5_output) if f.endswith(f'{settings["ranker"]}.{settings["metric"]}') and 'original' not in f]
            msmarco.aggregate(query_originals, query_changes, t5_output)

        box_path = join(t5_output, f'{settings["ranker"]}.{settings["metric"]}.datasets')
        if 'box' in settings['cmd']:
            if not os.path.isdir(box_path): os.makedirs(box_path)
            best_df = pd.read_csv(f'{t5_output}/{settings["ranker"]}.{settings["metric"]}.agg.best.tsv', sep='\t', header=0)
            qrels = pd.read_csv(f'{datapath}/qrels.train.nodups.tsv', names=['qid', 'did', 'pid', 'rel'], sep='\t')
            msmarco.box(best_df, qrels, box_path)

        if 'stamp' in settings['cmd']:
            print(f'Stamping diamond queries for {settings["ranker"]}.{settings["metric"]} == 1 ...')
            from evl import trecw
            if not os.path.isdir(join(t5_output,'runs')): os.makedirs(join(t5_output, 'runs'))

            diamond_initial = pd.read_csv(f'{box_path}/diamond.original.tsv', sep='\t', encoding='utf-8', index_col=False, header=None, names=['qid', 'query'], dtype={'qid': str})
            diamond_initial.drop_duplicates(subset=['qid'], inplace=True)#See msmarco.boxing(): in case we store more than two golden changes with same metric value
            msmarco.to_search_df(pd.DataFrame(diamond_initial['query']), f'{t5_output}/runs/diamond.original.{settings["ranker"]}', diamond_initial['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
            trecw.evaluate(f'{t5_output}/runs/diamond.original.{settings["ranker"]}', f'{t5_output}/runs/diamond.original.{settings["ranker"]}.{settings["metric"]}', qrels=f'{datapath}/qrels.train.nodups.tsv', metric=settings['metric'], lib=settings['treclib'])

            diamond_target = pd.read_csv(f'{box_path}/diamond.change.tsv', sep='\t', encoding='utf-8', index_col=False,header=None, names=['qid', 'query'], dtype={'qid': str})
            diamond_target.drop_duplicates(subset=['qid'], inplace=True)#See msmarco.boxing(): in case we store more than two golden changes with same metric value
            msmarco.to_search_df(pd.DataFrame(diamond_target['query']), f'{t5_output}/runs/diamond.change.{settings["ranker"]}', diamond_target['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
            trecw.evaluate(f'{t5_output}/runs/diamond.change.{settings["ranker"]}', f'{t5_output}/runs/diamond.change.{settings["ranker"]}.{settings["metric"]}', qrels=f'{datapath}/qrels.train.nodups.tsv', metric=settings['metric'], lib=settings['treclib'])

    if 'aol' in domain_list:
        datapath = data_list[domain_list.index('aol')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_output): os.makedirs(prep_output)
        index_item = '.'.join([item for item in settings['aol']['index_item']])
        in_type, out_type = settings['aol']['pairing'][1], settings['aol']['pairing'][2]
        tsv_path = {'train': f'{prep_output}/{in_type}.{out_type}.{index_item}.train.tsv', 'test': f'{prep_output}/{in_type}.{out_type}.{index_item}.test.tsv'}

        # AOL requires us to construct the Index, Qrels and Queries file from IR_dataset
        from dal import aol
        prep_index = f'./../data/raw/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(os.path.join(prep_index, 'indexes', index_item)): os.makedirs(os.path.join(prep_index, 'indexes', index_item))
        aol.initiate_queries_qrels(prep_index)
        aol.create_json_collection(prep_index, index_item)
        create_index('aol', index_item, param.settings['ncore'])

        if 'pair' in settings['cmd']:
            cat = True if 'docs' in {in_type, out_type} else False
            print(f'Pairing queries and relevant passages for training set ...')
            query_qrel_doc = aol.to_pair(prep_index, f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.{index_item}.train.tsv', index_item, cat=cat)
            query_qrel_doc.to_csv(tsv_path['train'], sep='\t', encoding='utf-8', columns=[in_type, out_type], index=False, header=False)
            print(f'Pairing queries and relevant passages for test set ...')
            query_qrel_doc.to_csv(tsv_path['test'], sep='\t', encoding='utf-8', columns=[in_type, out_type], index=False, header=False)

        t5_model = settings['t5model']  # {"small", "base", "large", "3B", "11B"} cross {"local", "gc"}
        t5_output = f'../output/{os.path.split(datapath)[-1]}/t5.{t5_model}.{in_type}.{out_type}'
        if {'finetune', 'predict'} & set(settings['cmd']):
            from mdl import mt5w
            if 'finetune' in settings['cmd']:
                print(f"Finetuning {t5_model} for {settings['iter']} and storing the checkpoints at {t5_output} ...")
                mt5w.finetune(
                    tsv_path=tsv_path,
                    pretrained_dir=f'./../output/t5-data/pretrained_models/{t5_model.split(".")[0]}',
                    # "gs://t5-data/pretrained_models/{"small", "base", "large", "3B", "11B"}
                    steps=settings['iter'],
                    output=t5_output, task_name='aol_cf',
                    lseq=settings['aol']['lseq'],
                    nexamples=query_qrel_doc.shape[0] if query_qrel_doc is not None else None, in_type=in_type,
                    out_type=out_type, gcloud=False)

            if 'predict' in settings['cmd']:
                print(f"Predicting {settings['nchanges']} query changes using {t5_model} and storing the results at {t5_output} ...")
                mt5w.predict(
                    iter=settings['nchanges'],
                    split='test',
                    tsv_path=tsv_path,
                    output=t5_output,
                    lseq=settings['aol']['lseq'],
                    gcloud=False)

        if 'search' in settings['cmd']:
            query_originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.{index_item}.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            query_changes = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["ranker"]}')
                             for f in listdir(t5_output) if isfile(join(t5_output, f)) and f.startswith('pred.') and
                             settings['ranker'] not in f and len([f for x in range(0,12) if f'{f}.split_{x}.{settings["ranker"]}' not in listdir(t5_output)]) > 0]
            with multiprocessing.Pool(settings['ncore']) as p:
                p.starmap(partial(aol.to_search, qids=query_originals['qid'].values.tolist(), index_item=index_item, ranker=settings['ranker'], topk=settings['topk'], batch=settings['batch']), query_changes)

            if not isfile(join(t5_output, f'original.{settings["ranker"]}')):
                query_originals.to_csv(f'{t5_output}/original', columns=['query'], index=False, header=False)
                aol.to_search_df(pd.DataFrame(query_originals['query']), f'{t5_output}/original.{settings["ranker"]}', query_originals['qid'].values.tolist(), index_item, settings['ranker'], topk=settings['topk'], batch=settings['batch'])

        if 'eval' in settings['cmd']:
            from evl import trecw
            search_results = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["metric"]}')
                              for f in listdir(t5_output) if isfile(join(t5_output, f)) and f.endswith(settings['ranker']) and
                              f'{f}.{settings["ranker"]}.{settings["metrics"]}' not in listdir(t5_output) for x in range(0,11)]

            with multiprocessing.Pool(settings['ncore']) as p: p.starmap(partial(trecw.evaluate, qrels=f'{datapath}/qrels.{index_item}.clean.tsv', metric=settings['metric'], lib=settings['treclib']), search_results)

    if ('yandex' in data_list): print('processing yandex...')


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -data ./../data/raw/toy.msmarco.passage)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -domain msmarco.passage)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='../output/', help='The output path (default: -output ../output/)')


# python -u main.py -data ../data/raw/toy.msmarco.passage -domain msmarco.passage

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='Personalized Query Refinement')
    addargs(parser)
    args = parser.parse_args()

    run(data_list=args.data_list,
        domain_list=args.domain_list,
        output=args.output,
        settings=param.settings)
