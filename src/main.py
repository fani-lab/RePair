import argparse, os, pandas as pd, multiprocessing
from functools import partial
from multiprocessing import freeze_support
from os import listdir
from os.path import isfile, join
from cmn.create_index import create_index
import param
from mdl import mt5w

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    if ('msmarco.passage' in domain_list):

        from dal import msmarco
        # seems the LuceneSearcher cannot be shared in multiple processes! See dal.msmarco.py

        datapath = data_list[domain_list.index('msmarco.passage')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_output): os.makedirs(prep_output)
        in_type, out_type = settings['msmarco.passage']['pairing'][1], settings['msmarco.passage']['pairing'][2]
        tsv_path = {'train': f'{prep_output}/{in_type}.{out_type}.train.tsv', 'test': f'{prep_output}/{in_type}.{out_type}.test.tsv'}

        query_qrel_doc = None
        if any(not os.path.exists(v) for k, v in tsv_path.items()):
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
        if 'finetune' in settings['cmd']:
            mt5w.finetune(
                tsv_path=tsv_path,
                pretrained_dir=f'./../output/t5-data/pretrained_models/{t5_model.split(".")[0]}', #"gs://t5-data/pretrained_models/{"small", "base", "large", "3B", "11B"}
                steps=5,
                output=t5_output, task_name='msmarco_passage_cf',
                lseq={"inputs": 32, "targets": 256},  #query length and doc length
                nexamples=query_qrel_doc.shape[0] if query_qrel_doc is not None else None, in_type=in_type, out_type=out_type, gcloud=False)

        if 'predict' in settings['cmd']:
            mt5w.predict(
                iter=5,
                split='test',
                tsv_path=tsv_path,
                output=t5_output,
                lseq={"inputs": 32, "targets": 256}, gcloud=False)

        if 'search' in settings['cmd']:
            #seems for some queries there is no qrels, so they are missed for t5 prediction.
            #query_originals = pd.read_csv(f'{datapath}/queries.train.tsv', sep='\t', names=['qid', 'query'], dtype={'qid': str})

            #we use the file after panda.merge that create the training set so we make sure the mapping of qids
            query_originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if "docs" in {in_type, out_type} else ""}.ctx.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            query_changes = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["ranker"]}') for f in listdir(t5_output) if isfile(join(t5_output, f)) and f.startswith('pred.') and settings['ranker'] not in f and f'{f}.{settings["ranker"]}' not in listdir(t5_output)]
            # for (i, o) in query_changes: msmarco.to_search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=100, batch=None)
            # batch search: for (i, o) in query_changes: msmarco.to_search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=100, batch=2)
            # parallel on each file
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                p.starmap(partial(msmarco.to_search, qids=query_originals['qid'].values.tolist(), ranker=settings['ranker'], topk=100, batch=None), query_changes)

            # we need to add the original queries as well
            if not isfile(join(t5_output, f'original.{settings["ranker"]}')):
                msmarco.to_search_df(pd.DataFrame(query_originals['query']), f'{t5_output}/original.{settings["ranker"]}', query_originals['qid'].values.tolist(), settings['ranker'], topk=100, batch=None)


        if 'eval' in settings['cmd']:
            from evl import trecw
            search_results = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["metric"]}') for f in listdir(t5_output) if isfile(join(t5_output, f)) and f.endswith(settings['ranker']) and f'{f}.{settings["ranker"].settings["metric"]}' not in listdir(t5_output)]

            # for (i, o) in search_results: trecw.evaluate(i, o, qrels=f'{datapath}/qrels.train.tsv', metric=settings['metric'], lib=settings['treclib'])
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                p.starmap(partial(trecw.evaluate, qrels=f'{datapath}/qrels.train.nodups.tsv', metric=settings['metric'], lib=settings['treclib']), search_results)
        if 'aggregate' in settings['cmd']:
            query_originals = pd.read_csv(
                f'{prep_output}/queries.qrels.doc{"s" if "docs" in {in_type, out_type} else ""}.ctx.train.tsv',
                sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            original_map = pd.read_csv(join(t5_output, 'original.bm25.map'), sep='\t', names=['map', 'qid', 'og_map'], index_col=False, low_memory=False)
            original_map.drop(columns='map', inplace=True)
            original_map.drop(original_map.tail(1).index, inplace=True)
            query_originals = query_originals.merge(original_map, how='left', on='qid')
            list_of_files = [('.'.join(f.split('.')[0:2]), f) for f in os.listdir(t5_output) if
                             isfile(join(t5_output, f)) and f.endswith('map') and f.startswith('pred')]
            msmarco.aggregate(query_originals, list_of_files, t5_output)
        if 'create_ds' in settings['cmd']:
            if not os.path.isdir(join(t5_output,'queries')): os.makedirs(join(t5_output,'queries'))
            if not os.path.isdir(join(t5_output, 'qrels')): os.makedirs(join(t5_output, 'qrels'))
            best_df = pd.read_csv(f'{t5_output}/bm25.map.agg.best.tsv', sep='\t', header=0)
            qrels = pd.read_csv(f'{datapath}/qrels.train.tsv',names=['qid','did','pid','rel'],sep='\t')
            msmarco.create_dataset(best_df,qrels,t5_output)

    if ('aol' in domain_list):
        datapath = data_list[domain_list.index('aol')]
        prep_index = f'./../data/raw/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_index): os.makedirs(prep_index)
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_output): os.makedirs(prep_output)
        index_item = settings['aol']['index_item'][0] if len(settings['aol']['index_item']) == 1 else '_'.join([item for item in settings['aol']['index_item']])
        in_type, out_type = settings['aol']['pairing'][1], settings['aol']['pairing'][2]
        tsv_path = {'train': f'{prep_output}/{in_type}.{out_type}.{index_item}.train.tsv',
                    'test': f'{prep_output}/{in_type}.{out_type}.{index_item}.test.tsv'}

        if not os.path.isdir(os.path.join(prep_index, 'indexes', index_item)): os.makedirs(os.path.join(prep_index, 'indexes', index_item))
        cat = True if 'docs' in {in_type, out_type} else False
        from dal import aol
        # AOL requires us to construct the Index, Qrels and Queries file from IR_dataset
        # create queries and qrels file
        aol.initiate_queries_qrels(prep_index)
        aol.create_json_collection(prep_index, index_item)
        create_index('aol', index_item)
        #to pair function
        if any(not os.path.exists(v) for k, v in tsv_path.items()):
            query_qrel_doc = aol.to_pair(prep_index, f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.{index_item}.train.tsv', index_item,
                                             cat=cat)
            query_qrel_doc.to_csv(tsv_path['train'], sep='\t', encoding='utf-8', columns=[in_type, out_type],
                                  index=False, header=False)
            query_qrel_doc.to_csv(tsv_path['test'], sep='\t', encoding='utf-8', columns=[in_type, out_type],
                                  index=False, header=False)
        t5_model = settings['t5model']  # {"small", "base", "large", "3B", "11B"} cross {"local", "gc"}
        t5_output = f'../output/{os.path.split(datapath)[-1]}/t5.{t5_model}.{index_item}.{in_type}.{out_type}'
        if not os.path.isdir(t5_output): os.makedirs(t5_output)

        if 'search' in settings['cmd']:
            query_originals = pd.read_csv(f'{prep_output}/queries.qrels.doc{"s" if cat else ""}.ctx.{index_item}.train.tsv', sep='\t', usecols=['qid', 'query'], dtype={'qid': str})
            query_changes = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["ranker"]}') for f in listdir(t5_output)
                             if isfile(join(t5_output, f)) and f.startswith('pred.') and settings[
                                 'ranker'] not in f and len([f for x in range(0,12) if f'{f}.split_{x}.{settings["ranker"]}' not in listdir(t5_output)]) > 0]
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                p.starmap(partial(aol.to_search, qids=query_originals['qid'].values.tolist(), index_item=index_item, ranker=settings['ranker'], topk=100, batch=None), query_changes)
            aol.to_search_df(pd.DataFrame(query_originals['query']), f'{t5_output}/original.{settings["ranker"]}', query_originals['qid'].values.tolist(), index_item, settings['ranker'], topk=100, batch=None)

        if 'eval' in settings['cmd']:
            from evl import trecw
            search_results = [(f'{t5_output}/{f}', f'{t5_output}/{f}.{settings["metric"]}') for f in listdir(t5_output) if
                              isfile(join(t5_output, f)) and f.endswith(settings['ranker']) and f'{f}.{settings["ranker"]}.{settings["metrics"]}' not in listdir(t5_output) for x in range(0,11)]

            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                p.starmap(partial(trecw.evaluate, qrels=f'{datapath}/qrels.{index_item}.clean.tsv', metric=settings['metric'],
                                  lib=settings['treclib']), search_results)
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
