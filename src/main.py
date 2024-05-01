import param
from os import listdir
from shutil import copyfile
from functools import partial
from itertools import product
from os.path import isfile, join, exists
from multiprocessing import freeze_support
from refinement import refiner_factory as rf
import argparse, os, pandas as pd, multiprocessing as mp


def hex_to_ansi(hex_color_code="", reset=False):
    #TODO: create a utlis for customizaion
    if reset: return "\033[0m"
    hex_color_code = hex_color_code.lstrip('#')
    red = int(hex_color_code[0:2], 16)
    green = int(hex_color_code[2:4], 16)
    blue = int(hex_color_code[4:6], 16)
    return f'\033[38;2;{red};{green};{blue}m'


def run(data_list, domain_list, output_result, corpora, settings):
    for domain in domain_list:
        print('-' * 50, f'Executing the project with the {hex_to_ansi("#BB8FCE")}{domain} {hex_to_ansi(reset=True)}dataset', '-' * 50)
        datapath = data_list[domain_list.index(domain)]

        if domain == 'msmarco.passage':
            from dal.msmarco import MsMarcoPsg
            ds = MsMarcoPsg(corpora[domain], domain)
        elif domain == 'aol-ia':
            from dal.aol import Aol
            ds = Aol(corpora[domain], domain, datapath, param.settings['ncore'])
        elif domain == 'yandex': raise ValueError('Yandex is yet to be added ...')
        else:
            from dal.ds import Dataset
            ds = Dataset(corpora[domain], domain)
        if 'trec' in corpora[domain]:
            [ds.read_queries(datapath, t, trec=True) for t in corpora[domain]['trec']]
            all_qrels = [pd.read_csv(os.path.join(datapath, f), sep='\t', index_col=False, names=ds.queries[0].qrel.keys()) for f in os.listdir(datapath) if f.endswith('train.tsv_') and f != 'qrels.train.tsv_']
            pd.concat(all_qrels, ignore_index=True).to_csv(f'{datapath}/qrels.train.tsv_', index=False, sep='\t', header=False)
        else: ds.read_queries(datapath, domain)

        refined_data_output = f'{output_result}{os.path.split(datapath)[-1]}'

        qrel_path = f'{datapath}/qrels.train.tsv_'
        copyfile('./param.py', f'{refined_data_output}/refiner_param.py')

        # Query refinement - refining queries using the selected refiners
        if 'query_refinement' in settings['cmd']:
            refiners = rf.get_nrf_refiner()
            if rf: refiners += rf.get_rf_refiner(output=refined_data_output, corpus=corpora[domain], ext_corpus=corpora[corpora[domain]['extcorpus']], ds=ds, domain=domain)
            with mp.Pool(settings['ncore']) as p:
                for refiner in refiners:
                    if refiner.get_model_name() == 'original': refiner_outfile = f'{refined_data_output}/{refiner.get_model_name()}'
                    if 't5' in refiner.get_model_name(): refiner.get_refined_query(""); continue
                    else: refiner_outfile = f'{refined_data_output}/refiner.{refiner.get_model_name()}'
                    if not exists(refiner_outfile):
                        ds.queries = p.map(partial(refiner.preprocess_query), ds.queries)
                        print(f'Writing results from {refiner.get_model_name()} queries in {refiner_outfile}')
                        refiner.write_queries(queries=ds.queries, outfile=refiner_outfile)
                    else: print(f'Results from {refiner.get_model_name()} queries in {refiner_outfile}')

        if 'similarity' in settings['cmd']:
            from evl import trecw
            if not os.path.isdir(f'{refined_data_output}/similarity/'): os.makedirs(f'{refined_data_output}/similarity/')
            search_results = [(f'{refined_data_output}/{f}', f'{refined_data_output}/similarity/{f}.similarity.csv') for f in listdir(refined_data_output) if f.startswith('refiner') and f.__contains__('bt') and f'{f}.{metric}' not in listdir(output)]
            with mp.Pool(settings['ncore']) as p: p.starmap(partial(trecw.compare_query_similarity), search_results)

        if any(item in ['search', 'rag_fusion', 'eval', 'agg', 'box'] for item in settings['cmd']):
            for ranker, metric in product(param.settings['ranker'], param.settings['metric']):
                print('-' * 30, f'Ranking and evaluating by {hex_to_ansi("#3498DB")}{ranker}{hex_to_ansi(reset=True)} and {hex_to_ansi("#3498DB")}{metric}{hex_to_ansi(reset=True)}')
                output = f'{refined_data_output}/{ranker}.{metric}'
                if not os.path.isdir(output): os.makedirs(output)

                if 'search' in settings['cmd']:  # 'bm25 ranker'
                    print(f"Searching documents for query changes using {ranker} ...")
                    # Considers generated queries from t5 or refiners and the original queries
                    query_changes = [(f'{refined_data_output}/{f}', f'{output}/{f}.{ranker}') for f in listdir(refined_data_output) if isfile(join(refined_data_output, f)) and ((f.startswith('pred.') and len(f.split('.')) == 2) or (f.startswith('refiner.') or f.startswith('original')) and f'{f}.{ranker}' not in listdir(output))]
                    # Seems the LuceneSearcher cannot be shared in multiple processes! All the variables in class cannot be shared!
                    with mp.Pool(settings['ncore']) as p: p.starmap(partial(ds.search, qids=[query.qid for query in ds.queries], ranker=ranker, topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'], index=ds.settings["index"], settings=corpora[domain]), query_changes)

                if 'rag_fusion' in settings['cmd']:
                    from evl import trecw
                    print('RAG Fusion Step ...')
                    columns = ['id', 'Q0', 'doc', 'rank', 'score', 'Pyserini']
                    for categorize in settings['fusion']:
                        if any(file.startswith(f'rag_fusion.{categorize}.{ranker}') for file in os.listdir(output)): continue
                        print(f'RAG Fusion for {hex_to_ansi("#3498DB")}"{categorize}"{hex_to_ansi(reset=True)} category')
                        names = ds.get_refiner_list(categorize)
                        mrr_results = pd.concat([pd.read_csv(os.path.join(output, f), sep='\t', names=columns).assign(refiner=('original' if 'original' in f else ('.'.join(f.split('.')[1:3]) if 'stem' in f else f.split('.')[1]))) for f in os.listdir(output) if f.endswith(ranker) and not f.startswith('rag_fusion') and any(name in f for name in names)], ignore_index=True)
                        mrr_results = mrr_results.groupby(['id', 'doc'])
                        ds.reciprocal_rank_fusion(docs=mrr_results, k=0, columns=columns, output=f'{output}/rag_fusion.{categorize}.{ranker}')

                if 'eval' in settings['cmd']:
                    from evl import trecw
                    print(f'Evaluating documents for query changes using {hex_to_ansi("#3498DB")}{metric}{hex_to_ansi(reset=True)} ...')
                    search_results = [(f'{output}/{f}', f'{output}/{f}.{metric}') for f in listdir(output) if f.endswith(ranker) and f'{f}.{metric}' not in listdir(output)]
                    with mp.Pool(settings['ncore']) as p: p.starmap(partial(trecw.evaluate, qrels=qrel_path, metric=metric, lib=settings['treclib'], mean=not settings['large_ds']), search_results)

                if 'agg' in settings['cmd']:
                    originals = pd.DataFrame({'qid': [str(query.qid) for query in ds.queries], 'query': [query.q for query in ds.queries]})
                    original_metric_values = pd.read_csv(join(output, f'original.{ranker}.{metric}'), sep='\t', usecols=[1, 2], names=['qid', f'original.{ranker}.{metric}'], index_col=False, dtype={'qid': str})

                    originals = originals.merge(original_metric_values, how='left', on='qid')
                    originals[f'original.{ranker}.{metric}'].fillna(0, inplace=True)
                    changes = [(f.split(f'.{ranker}.{metric}')[0], f) for f in os.listdir(output) if f.endswith(f'{ranker}.{metric}') and f.startswith('refiner')]
                    print(f'Aggregating results for all {hex_to_ansi("#3498DB")}refiners{hex_to_ansi(reset=True)} ...')
                    ds.aggregate(originals, refined_data_output, changes, output)
                    if 'rag' in settings['cmd']:
                        changes = [f for f in os.listdir(output) if f.endswith(f'{ranker}.{metric}') and (f.startswith('refiner') or f.startswith('rag_fusion'))]
                        print(f'Aggregating results for all {hex_to_ansi("#3498DB")}refiners{hex_to_ansi(reset=True)} and {hex_to_ansi("#3498DB")}rag_fusion{hex_to_ansi(reset=True)} ...')
                        ds.aggregate_refiner_rag(original_metric_values, changes, output)

                if 'box' in settings['cmd']:
                    from evl import trecw
                    box_path = join(output, f'{ranker}.{metric}.boxes')
                    if not os.path.isdir(box_path): os.makedirs(box_path)
                    gold_df = pd.read_csv(f'{output}/{ranker}.{metric}.agg.all.tsv', sep='\t', header=0, dtype={'qid': str})

                    qrels_list = [pd.DataFrame(query.qrel) for query in ds.queries]
                    qrels = pd.concat(qrels_list, ignore_index=True)

                    box_condition = settings['box']
                    ds.box(gold_df, qrels, box_path, box_condition)
                    for c in box_condition.keys():
                        print(f'{c}: Stamping boxes for {ranker}.{metric} before and after refinements ...')
                        if not os.path.isdir(join(box_path, 'stamps')): os.makedirs(join(box_path, 'stamps'))
                        df = pd.read_csv(f'{box_path}/{c}.tsv', sep='\t', encoding='utf-8', index_col=False, header=None, names=['qid', 'query', 'metric', 'query_', 'metric_'], dtype={'qid': str})
                        df.drop_duplicates(subset=['qid'], inplace=True)  # See ds.boxing(): in case we store more than two changes with the same metric value
                        if df['query'].to_frame().empty: print(f'No queries for {c}')
                        else:
                            ds.search(df['query'].to_frame(), f'{box_path}/stamps/{c}.original.{ranker}', df['qid'].values.tolist(),ranker, topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'], index=ds.settings["index"], settings=corpora[domain])
                            trecw.evaluate(f'{box_path}/stamps/{c}.original.{ranker}', f'{box_path}/stamps/{c}.original.{ranker}.{metric}', qrels=qrel_path, metric=metric, lib=settings['treclib'], mean=True)
                            ds.search(df['query_'].to_frame().rename(columns={'query_': 'query'}), f'{box_path}/stamps/{c}.change.{ranker}', df['qid'].values.tolist(),ranker, topk=settings['topk'], batch=settings['batch'], ncores=settings['ncore'], index=ds.settings["index"], settings=corpora[domain])
                            trecw.evaluate(f'{box_path}/stamps/{c}.change.{ranker}', f'{box_path}/stamps/{c}.change.{ranker}.{metric}', qrels=qrel_path, metric=metric, lib=settings['treclib'], mean=True)

                if 'dense_retrieve' in settings['cmd']:
                    from evl import trecw
                    from tqdm import tqdm
                    condition = 'no_pred'
                    if not isfile(join(output, f'{ranker}.{metric}.agg.{condition}.tsv')):
                        agg_df = pd.read_csv(f'{output}/{ranker}.{metric}.agg.all_.tsv', sep='\t', header=0, dtype={'qid': str})
                        changes = [(f, f'{f}.{ranker}.{metric}') for f in os.listdir(output) if f.startswith('pred') and len(f.split('.')) == 2]
                        # creates a new file for poor performing/no prediction queries
                        with open(f'{output}/{ranker}.{metric}.agg.{condition}.tsv', mode='w', encoding='UTF-8') as agg_poor_perf:
                            agg_poor_perf.write(f'qid\tquery\t{ranker}.{metric}\t\tquery_\t{ranker}.{metric}_\n')
                            for index, row in tqdm(agg_df.iterrows(), total=agg_df.shape[0]):
                                all = list()
                                for change, metric_value in changes: all.append((row[change], row[f'{change}.{ranker}.{metric}'], change))
                                all = sorted(all, key=lambda x: x[1], reverse=True)
                                # if row[f'original.{ranker}.{metric}'] == 0 and all[0][1] <= 0.1: #poor perf
                                if row[f'original.{ranker}.{metric}'] > all[0][1] and row[f'original.{ranker}.{metric}'] <= 1:  # no prediction
                                    agg_poor_perf.write(f'{row.qid}\t{row.query}\t{row[f"original.{ranker}.{metric}"]}\t{all[0][0]}\t{all[0][1]}\n')
                    original = pd.read_csv(f'{output}/{ranker}.{metric}.agg.{condition}.tsv', sep='\t', encoding="utf-8", header=0, index_col=False, names=['qid', 'query', f'{ranker}.{metric}', 'query_', f'{ranker}.{metric}_'])
                    if domain == 'aol-ia':
                        original['pid'] = original['pid'].astype('str')
                        original = original[:16230]
                    print(original.shape[0])
                    pred = pd.DataFrame()
                    pred["query"] = original["query_"]
                    search_list = [(pd.DataFrame(original['query']), f'{output}/original.{condition}.tct_colbert'),
                                   (pd.DataFrame(pred['query']), f'{output}/pred.{condition}.tct_colbert')]
                    search_results = [(f'{output}/original.{condition}.tct_colbert', f'{output}/original.{condition}.tct_colbert.{metric}'),
                                      (f'{output}/pred.{condition}.tct_colbert', f'{output}/pred.{condition}.tct_colbert.{metric}')]
                    with mp.Pool(settings['ncore']) as p:
                        p.starmap(partial(ds.search_list, qids=original['qid'].values.tolist(), ranker='tct_colbert', topk=100, batch=None, ncores=settings['ncore'], index=settings[f'{domain}']["dense_index"], encoder=settings[f'{domain}']['dense_encoder']), search_list)
                        p.starmap(partial(trecw.evaluate, qrels=qrel_path, metric=metric, lib=settings['treclib']), search_results)

                    # aggregate colbert results and compare with bm25 results
                    original_dense = pd.read_csv(f'{output}/original.{condition}.tct_colbert.{metric}', sep='\t', usecols=[1, 2], names=['qid', f'{metric}'])
                    pred_dense = pd.read_csv(f'{output}/pred.{condition}.tct_colbert.{metric}', sep='\t', usecols=[1, 2], names=['qid', f'{metric}'])
                    agg_df = original_dense.merge(pred_dense, on='qid', suffixes=('', '_'), how='outer')
                    print(f"total queries: {agg_df.shape[0]}")
                    print(f"pred greater than original with 0 :{agg_df[(agg_df[f'{metric}'] == 0) & (agg_df[f'{metric}_'] > 0)].shape[0]}")  # where original = 0 and pred > original
                    print(f"pred greater than original {agg_df[agg_df[f'{metric}_'] > agg_df[f'{metric}']].shape[0]}")  # pred > original
                    print(f"pred less than original {agg_df[agg_df[f'{metric}_'] < agg_df[f'{metric}']].shape[0]}")  # pred < original

                    print(f"original sparse:{original[f'{ranker}.{metric}'].mean()}\n")
                    print(f"original dense:{agg_df[f'{metric}'].mean()}\n")
                    print(f"pred sparse:{original[f'{ranker}.{metric}_'].mean()}")
                    print(f"pred dense:{agg_df[f'{metric}_'].mean()}\n")
                    # colbert improvements
                    agg_df['original_sparse'] = original[f'{ranker}.{metric}']
                    agg_df['pred_sparse'] = original[f'{ranker}.{metric}_']
                    agg_df.to_csv(f'{output}/colbert.comparison.{condition}.{metric}.tsv', sep="\t", index=None)

                if 'stats' in settings['cmd']: from stats import stats


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=param.settings['datalist'], help='a list of dataset paths; required; (eg. -data ./../data/raw/toy.msmarco.passage)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str, default=param.settings['domainlist'], help='a list of dataset paths; required; (eg. -domain msmarco.passage)')

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

    run(data_list=args.data_list,
        domain_list=args.domain_list,
        output_result=args.output,
        corpora=param.corpora,
        settings=param.settings)
