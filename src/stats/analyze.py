from src.refinement.refiner_param import refiners
from src.refinement.lang_code import other, nllb
import matplotlib.pyplot as plt
from src.param import settings
from itertools import product
import pandas as pd
import shutil
import json
import sys
import os
import re

# sys.path.extend(["./output"])


def get_refiner_list(category):
    names = list(refiners[category].keys())
    if 'BackTranslation' in names:
        names.remove('BackTranslation')
        names.append('bt_nllb_')
        names.append('bt_bing_')
    return [re.sub(r'\b(\w+)Stemmer\b', r'stem.\1', re.sub(r'\b\w*BackTranslation\w*\b', 'bt', name)).lower() for name in names]


def combine_refiner_results(infile, output, datasets, ranker_metrics):
    """
    Combine the results of refiner comparison for multiple datasets and ranker-metric combinations.

    Parameters:
    - datasets (list): List of dataset names.
    - ranker_metrics (list of tuples): List of tuples containing ranker and metric combinations.

    Returns:
    - None: Writes the combined results to CSV files for each ranker-metric combination.

    Description:
    This function combines the results of refiner comparison for multiple datasets and ranker-metric combinations.
    It reads the comparison results for each dataset and ranker-metric combination from CSV files, merges them based
    on the 'refiner' column, and writes the combined results to CSV files for each ranker-metric combination.
    """
    for r, m in ranker_metrics:
        # df_list = [pd.read_csv(f'{infile}.{ds}.{r}.{m}.csv', skiprows=1, names=['category', 'refiner', f'{ds}_|q**|', f'{ds}_%']) for ds in datasets]
        df_list = [pd.read_csv(f'{infile}/{f}', skiprows=1, names=['category', 'refiner', f'{f.split(".")[0]}_|q**|', f'{f.split(".")[0]}_%']) for f in os.listdir(infile) if f'{r}.{m}' in f]
        merged_df = df_list[0]
        for df in df_list[1:]: merged_df = pd.merge(merged_df, df, on=['refiner', 'category'], how='outer')
        merged_df = merged_df.fillna(0)
        merged_df.to_csv(f'{output}.{r}.{m}.csv', index=False)


def compare_refiners(infile, output, globalr, ranker, metric, refiners_list=[], overlap=False, agg=False):
    """
    Compare the performance of different query refiners.

    Parameters:
    - infile (str): Path to the input file containing refiner data.
    - output (str): Path to the output file where the comparison results will be saved.
    - globalr (list): List of refiner names indicating global refiners.
    - ranker (str): Ranker method.
    - metric (str): Evaluation metric.
    - overlap (bool): Flag indicating whether to consider the best refined queries among refiners or the all the refined queries. Default is False.
    - agg_bt (bool): Flag indicating whether to aggregate backtranslation (bt) results together or consider the bt_lang. Default is False.

    Returns:
    - None: Writes the comparison results to the output file.

    Description:
    This function reads refiner data from an input file, preprocesses it, computes the percentage of queries processed by each refiner,
    and categorizes refiners as global or local based on provided lists for global refiners. If specified, backtranslation data can be aggregated.
    The comparison results are then sorted, saved to an output file, and categorized as global or local.
    """
    df = pd.read_csv(infile, sep='\t', header=0)
    num_q = df['qid'].nunique()  # Number of queries in the dataset
    # if len(refiners_list) == 0: refiners_list = get_refiner_list('global') + get_refiner_list('local') + ['-1'] + ['msmarco', 'base'] + [f"rag.{second}.{third}" for second in ['all', 'bt_nllb', 'bt', 'global', 'local'] for third in ['k60', 'k0']]
    if agg:
        filtered_dfs = []
        # for refiner_name in refiners_list:
        #     filtered_df = df.loc[df[df['order'].str.contains(refiner_name)].groupby('qid')[f'{ranker}.{metric}'].idxmax()]
        #     filtered_df['order'] = refiner_name
        #     filtered_dfs.append(filtered_df)
        # df = pd.concat(filtered_dfs, ignore_index=True)
        df = df.loc[df.groupby('qid')[f'{ranker}.{metric}'].idxmax()]
        df.reset_index(drop=True, inplace=True)

    # df = df[df['order'].str.contains('|'.join(refiners_list))]
    # Selecting q** (The best query among all refined queries for each qid)
    if not overlap:
        max_indices = df.groupby('qid')[f'{ranker}.{metric}'].idxmax()
        df = df.loc[max_indices]

    df.reset_index(drop=True, inplace=True)
    # plot_chart(df, ranker, metric, f'{output}.png')
    # df['order'] = df['order'].apply(lambda x: 'q^' if '-1' in x else x)
    df['order'] = df['order'].apply(lambda x: 'original' if '-1' in x else x)

    # Write results in a file
    final_table = pd.DataFrame(columns=['category', 'refiner', '|q**|', '%'])
    for refiner, group in df.groupby('order'):
        if 'rag' in refiner: c = 'rag'
        elif 't5' in refiner: c = 'llm'
        elif 'original' in refiner: c = 'original'
        else: c = 'global 'if any(name in refiner for name in globalr) else 'local'
        if refiner.startswith('bt_'):
            refiner = refiner[:-1]
        final_table.loc[len(final_table)] = [c, refiner, len(group), (len(group)/num_q)*100]
    final_table = final_table.sort_values(by=['category', 'refiner'], ascending=[True, True])
    final_table.to_csv(f'{output}', index=False)

def count_query_len():
    is_tag_file = False
    q = ''
    df = pd.DataFrame(columns=["dataset", "|q|", "|len(q)|"])
    for domain in settings['domainlist']:
        original = pd.read_csv(f'./output/{domain}/refiner.original', sep='\t', usecols=[2], names=['q'], skip_blank_lines=False, engine='python', index_col=False, header=None)
        lengths = original['q'].apply(lambda x: len(x.split(' ')))
        df.loc[len(df)] = [domain, len(lengths), lengths.mean()]
    df.to_csv(f'./output/analyze/avg_len_query.all.csv', index=False)


def create_analyze_table(refiner_name):
    """
    Automates the analysis and generation of a tabular summary based on aggregated data from different datasets,
    rankers, and metrics, with respect to a specified refiner.

    Parameters:
    -----------
    refiner_name : str
        The name of the refiner to be analyzed.

    Returns:
    --------
    None

    Output:
    -------
    Generates and saves a CSV file containing the analyzed table with the following columns:
        - dataset: Name of the dataset
        - |Q|: Number of queries in the dataset
        - |Q'|: Number of queries that needs refinement
        - ir_metric: IR metric name
        - |Q*|: Number of refined queries
        - %: Percentage of refined queries compared to the total queries that need refinement
        - delta: Average improvement in the metric value for refined queries compared to the original query

    Example Usage:
    --------------
    create_analyze_table('bt')

    """
    final_table = pd.DataFrame(columns=["dataset", "|Q|", "|Q'|", "ir_metric", "|Q*|", "%", "delta"])
    for ds in ['dbpedia', 'robust04']: # ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']
        for ranker, metric in product(settings['ranker'], settings['metric']):
            # df = pd.read_csv(f'./output/{ds}/{ranker}.{metric}/{ranker}.{metric}.agg.platinum.tsv', sep='\t', header=0)
            df = pd.read_csv(f'./output/t5/{ds}/{ranker}.{metric}.agg.plat.tsv', sep='\t', header=0)
            df = df[df['order'].str.contains(refiner_name) | (df['order'] == '-1')]
            num_q = df['qid'].nunique()  # Number of queries in the dataset
            q_prim = len(df[(df['order'] == '-1') & (df[f'{ranker}.{metric}'] != 1)])
            q_star = df[df['order'].str.contains(refiner_name)].groupby('qid').ngroups
            percent = (q_star/q_prim)*100

            avg_metric = 0
            max_metric_index = df[df['order'].str.contains(refiner_name)].groupby('qid')[f'{ranker}.{metric}'].idxmax()
            for idx in max_metric_index: avg_metric += df.loc[idx, f'{ranker}.{metric}'] - df.loc[(df['qid'] == df.loc[idx, 'qid']) & (df['order'] == '-1'), f'{ranker}.{metric}'].values[0]
            avg_metric = 0 if q_star==0 else avg_metric / q_star

            extra_col = {}
            if 'bt' in refiner_name:
                lang = ['persian', 'french', 'german', 'russian', 'malay', 'tamil', 'swahili', 'chinese_simplified', 'korean', 'arabic']
                extra_col = {f'{item1}_{item2}': 0 for item1, item2 in product(lang, ['%', 'delta'])}
                for refiner, group in df.groupby('order'):
                    if refiner == '-1': continue
                    avg = group[f'{ranker}.{metric}'].sum()
                    qid_list = group['qid'].tolist()
                    avg_original = df[(df['qid'].isin(qid_list)) & (df['order'] == '-1')][f'{ranker}.{metric}'].sum()
                    extra_col[f'{refiner.replace(refiner_name+"_", "", 1)}_%'] = len(group)/q_prim * 100
                    extra_col[f'{refiner.replace(refiner_name+"_", "", 1)}_delta'] = f'+{(avg - avg_original)/len(group)}'

            new_line = {"dataset":ds, "|Q|":num_q, "|Q'|":q_prim, "ir_metric":f'{ranker}.{metric}', "|Q*|":q_star, "%":percent, "delta":f'+{avg_metric}', **extra_col}
            final_table = pd.concat([final_table, pd.DataFrame([new_line])], ignore_index=True)
    # final_table.to_csv(f'./output/analyze/analyze_{refiner_name}.all.csv', index=False)
    final_table.to_csv(f'./output/analyze/analyze_t5.all.csv', index=False)


def refiner_distribution_table(infile, output):
    for ds in ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']:
        for ranker, metric in product(settings['ranker'], settings['metric']):
            df = pd.read_csv(f'{infile}/{ds}/{ranker}.{metric}/{ranker}.{metric}.agg.all.tsv', sep='\t', header=0)
            df['delta'] = df['delta'] = df.groupby('qid')[f'{ranker}.{metric}'].transform(lambda x: x - x[df['order'] == '-1'].iloc[0])
            df.reset_index(drop=True, inplace=True)
            filtered_dfs = []
            # refiners_list = ['-1', 'bt_nllb', 'bt_bing', 'tagmee', 'relevancefeedback', 'anchor']
            refiners_list = ['-1', 'bt_nllb', 'bt_bing']
            for refiner_name in refiners_list:
                filtered_df = df.loc[
                    df[df['order'].str.contains(refiner_name)].groupby('qid')[f'{ranker}.{metric}'].idxmax()]
                filtered_df['order'] = refiner_name
                filtered_dfs.append(filtered_df)
            df = pd.concat(filtered_dfs, ignore_index=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(f'{output}/cal.delta.bt.original.{ds}.{ranker}.{metric}.csv', index=False)
            # plot_chart(df, ranker, metric, f'{output}/cal.delta.refiner.original.{ds}.{ranker}.{metric}.png')


def plot_chart(df, ranker, metric, output):
    colors = ["#C2FF3300", "#7800FF00", "#FFC30000", "#0082FF00", "#FF573300", "#00FFA700", "#0082FF00", "#FF00E500", "#FF007700", "#00FFCC00","#FFDC0000"]
    grouped_df = df.groupby('order')
    index = 0
    for order, group in grouped_df:
        if order == '-1': continue
        trend = group['delta'].to_list()
        # trend = sorted(trend, reverse=True)
        plt.hist(trend, label=order.split('.')[0], alpha=0.8)
        index += 1
    # plt.gca().axes.get_xaxis().set_visible(False)
    plt.xlabel(f'{ranker}.{metric}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'Scatter plot {ranker}.{metric}')
    plt.savefig(output)
    plt.close()
    # plt.show()


def get_predictions(infile, output, original):
    for baseline in ['acg', 'seq2seq', 'hredqs']:
        prediction = pd.read_json(f'{infile}/{baseline}/{baseline}.e100_test.json', lines=True)
        prediction['previous_queries'] = prediction['previous_queries'].apply(lambda x: ', '.join(x))
        prediction['session_id'] = prediction['session_id'].astype(str)
        prediction['session_id'] = prediction['session_id'].apply(lambda x: x[:-1])
        prediction['predictions'] = prediction['predictions'].apply(lambda x: ', '.join(x))

        reference = pd.read_csv(original, names=['session_id', 'nnn', 'references'], sep='\t', header=0)
        reference['session_id'] = reference['session_id'].astype(str)
        reference = reference.drop(columns=['nnn'])

        df = pd.merge(prediction, reference, on='session_id', how='inner', copy=False)
        new_df = df[['session_id', 'predictions']]
        new_df = new_df.sort_values(by='session_id')

        new_df.to_csv(f'{output}.{baseline}',  index=False, header=False, sep='\t')


def analyze_similarity(infile, output):
    result = pd.DataFrame(columns=['dataset', 'lang', '|ql|', 'semsim', 'rougeL'])
    for ds in ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']: # ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']
        df_list = [(f.split('.')[1].split('_')[2], pd.read_csv(f'{infile}/{ds}/similarity/{f}', usecols=['refined', 'rougeL', 'semsim'])) for f in os.listdir(f'{infile}/{ds}/similarity') if f.startswith('refiner.bt_bing')]
        for (lang, df) in df_list:
            df['len(refined)'] = df['refined'].apply(lambda x: len(str(x).split()))
            df = df.drop(columns=['refined'])
            averages = df.mean()
            result.loc[len(result)] = [ds, lang, averages['len(refined)'], averages['semsim'], averages['rougeL']]

    result.to_csv(f'{output}/similarity.bing.all.csv', index=False)


def get_qid(dataset):
    infile_bt = f'./output/{dataset}/bm25.map/bm25.map.agg.bt.platinum.tsv'
    infile_refiner = f'./output/{dataset}/bm25.map/bm25.map.agg.refiner.platinum.tsv'

    df_bt = pd.read_csv(infile_bt, sep='\t', usecols=[0, 1], names=['qid', 'order'], header=0)
    df_refiner = pd.read_csv(infile_refiner, sep='\t', usecols=[0, 1], names=['qid', 'order'], header=0)

    groups_bt = df_bt.groupby(['qid'])
    groups_refiner = df_refiner.groupby(['qid'])

    list_bt = []
    for qid, group in groups_bt:
        if len(group) == 1:
            list_bt.append(qid)

    list_refiner = []
    for qid, group in groups_refiner:
        if len(group) == 1:
            list_refiner.append(qid)

    qids =  set(list_bt) & set(list_refiner)
    return [str(x[0]) for x in qids]


def get_avg(ranker_list, metric_list, ds_list, output):
    headers = ['cat', 'ranker', 'metric', 'avg']
    for ds in ds_list:
        df_result = pd.DataFrame(columns=headers) #['dataset', 'bm25.map', 'bm25.ndcg', 'bm25.recip_rank', 'qld.map', 'qld.ndcg', 'qld.recip_rank']
        for ranker, metric in product(ranker_list, metric_list):
            inpath = f'./output/{ds}/{ranker}.{metric}'
            files = [f'{inpath}/original.{ranker}.{metric}']
            files.extend([(f'{inpath}/rag/fusion/{f}') for f in os.listdir(f'{inpath}/rag/fusion') if f.startswith('rag') and f.endswith(f'{ranker}.{metric}')])
            for f in files:
                df = pd.read_csv(f, sep='\t', usecols=[1, 2], names=['qid', metric], index_col=False, skipfooter=1, dtype={'qid': str}, engine='python')
                df_result.loc[len(df_result)] = [(f.split('/')[-1]).rsplit('.', 2)[0], ranker, metric, df[metric].mean()]
        df_result.to_csv(f'{output}/rag.vs.original.{ds}.csv', index=False)


def rename_files(folder_path, ranker, metric):
    def get_key_from_value(value):
        for key, val in nllb.items():
            if val.lower() == value:
                return key
        return None

    # Filter files that contain 'bt' but not 'bing'
    filtered_files = [f for f in os.listdir(folder_path) if f.startswith('rag_fusion') and os.path.isfile(os.path.join(folder_path, f))]

    # Rename files
    for file in filtered_files:
        # Split the filename into parts
        parts = file.split(f'rag_fusion')
        if len(parts) == 2:
            new_name = 'rag' + parts[1]
            # Rename the file
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
            print(f"Renamed {file} to {new_name}")


def move_files(source_dir, target_dir):
    """
    Args:
    source_dir (str): The path to the source directory.
    target_dir (str): The path to the target directory.
    """
    filtered_files = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    if not os.path.isdir(output): os.makedirs(output)
    for file in filtered_files:
        shutil.move(file, target_dir)
        print(f'Moved file: {file} from {source_dir} to {target_dir}')


def get_agg(infile, output, ranker, metric, refiners=['nllb', 'refiner', 'bt']):
    for ref in refiners:
        df = pd.read_csv(f'{infile}/{ranker}.{metric}/{ranker}.{metric}.agg.{ref}.platinum.tsv', sep='\t', header=0)
        max_indices = df.groupby('qid')[f'{ranker}.{metric}'].idxmax()
        df = df.loc[max_indices]
        df = df.drop(columns=['order', f'{ranker}.{metric}'])
        df.to_csv(f'{output}/{ref}.original', sep='\t', index=False)


def merge_plat_files(infiles, output, rm):
    final_df = pd.DataFrame()
    for i, file in enumerate(infiles):
        df = pd.read_csv(file, sep='\t', header=0, index_col=False, skipfooter=1, dtype={'qid': str})
        if i != 0: df = df[df['order'] != '-1']
        if 't5' in file:
            folder = 'base' if 'base' in file else 'msmarco'
            df['order'] = df['order'].apply(lambda x: '-1' if str(x) == '-1' else (f'{folder}.{".".join(x.split(".")[-2:])}') if file.split('/')[4] != 'original' else f'{folder}.{x.split(".")[-1]}')
        final_df = pd.concat([final_df, df], axis=0)
    final_df['qid'] = final_df['qid'].str.strip()
    final_df = final_df.sort_values(by=['qid', rm], ascending=[True, False])
    final_df.to_csv(output, sep='\t', index=False)


def get_plat_files(infile, ds, ranker, metric, llm=False):
    files = []
    if llm:
        # original t5 base
        files.extend([f'{infile}/t5/base/original/{ds}/{ranker}.{metric}.agg.plat.tsv'])
        # original t5 msmarco
        files.extend([f'{infile}/t5/finetuned.msmarco/original/{ds}/{ranker}.{metric}.agg.plat.tsv'])
        # predict t5 base
        files.extend([f'{infile}/t5/base/predict/{ds}/{ranker}.{metric}.agg.plat.tsv'])
        # predict t5 msmarco
        files.extend([f'{infile}/t5/finetuned.msmarco/predict/{ds}/{ranker}.{metric}.agg.plat.tsv'])
    else:
        # No rag - allref
        # files.extend([f'{infile}/{ds}/{ranker}.{metric}/agg/{ranker}.{metric}.agg.allref.platinum.tsv'])
        # rag - all category
        for category in ['allref', 'bt', 'global', 'local']:
            files.extend([f'{infile}/{ds}/{ranker}.{metric}/rag/fusion/{ranker}.{metric}.agg.{category}.platinum.tsv'])
    return files


def get_k_parameters(infile, output):
    if not os.path.isdir(output): os.makedirs(output)
    for ds in ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']:  # ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']
        df_result = pd.DataFrame(columns=['category', 'k', 'bm25.map', 'qld.map'])
        for category in ['bt', 'bt_nllb', 'global', 'local']:  # ['bt', 'bt_nllb', 'global', 'local']
            for k in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:  # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                dict = {'category':category, 'k':k}
                for ranker, metric in product(['bm25', 'qld'], ['map']):  # product(settings['ranker'], settings['metric'])
                    df_read = pd.read_csv(f'{infile}/{ds}/{ranker}.{metric}/rag/fusion/multi/rag.{category}.k{k}.{ranker}.{metric}', sep='\t', usecols=[1, 2], names=['qid', metric], index_col=False, skipfooter=1, dtype={'qid': str}, engine='python')
                    dict[f'{ranker}.{metric}'] = df_read[metric].mean()
                df_result.loc[len(df_result)] = dict
        df_result.to_csv(f'{output}/analyze/rag/{ds}.multi_k_results.csv', index=False)

            # df_read = pd.read_csv(f'./output/{ds}/{ranker}.{metric}/rag/fusion/multi/rag.{category}.k{k}.{ranker}', sep='\t', names=['qid', 'Q0', 'docid', 'rank', 'score', 'Pyserini'], index_col=False, skipfooter=1, dtype={'qid': str, 'docid': str, 'rank':str}, engine='python')
            # print(f'Reading rag.{category}.k{k}.{ranker}...')
            # df_read['docid'] = df_read['docid'].str.strip()
            # df_read = df_read.sort_values(by=['qid', 'docid', 'rank'])
            # mask = df_read['docid'] != df_read['docid'].shift()
            # df_read = df_read[mask]
            # df_read = df_read.sort_values(by=['qid', 'rank'])
            # df_read['qid_Q0'] = df_read['qid'] + ' ' + df_read['Q0']
            # df_read = df_read.drop(['qid', 'Q0'], axis=1)
            #
            # df_read['docid_rank'] = df_read['docid'] + ' ' + df_read['rank']
            # df_read = df_read.drop(['docid', 'rank'], axis=1)
            #
            # df_read = df_read.reindex(columns=['qid_Q0', 'docid_rank', 'score', 'Pyserini'])
            #
            # df_read.to_csv(f'./output/{ds}/{ranker}.{metric}/rag/fusion/multi/rag.{category}.k{k}.{ranker}', sep='\t', index=False, header=False)
            # print(f'Writing rag.{category}.k{k}.{ranker}...')

if __name__ == '__main__':
    globalr = get_refiner_list('global')
    # localr = get_refiner_list('local')
    # selected_refiner = 'refiner'
    infile = f'./output'
    output = f'./output/analyze/rag'
    ranker_list = ['bm25', 'qld']
    metric_list = ['map', 'ndcg', 'recip_rank']
    ds_list = ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b'] # ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']
    if not os.path.isdir(output): os.makedirs(output)
    # for ds in ds_list:
        # rename_files(f'{infile}')
        # [rename_files(f'{infile}/{ranker}.{metric}/rag', ranker, metric) for ranker, metric in product(settings['ranker'], settings['metric'])]
        # [move_files(f'{infile}/{ds}/{ranker}.{metric}/rag', f'{infile}/{ds}/{ranker}.{metric}/rag/fusion') for ranker, metric in product(ranker_list, metric_list)]
        # [compare_refiners(infile=f'{infile}/{ranker}.{metric}/agg/{ranker}.{metric}.agg.allref.platinum.tsv', output=f'{output}/compare.allref.{ds}.{ranker}.{metric}', globalr=globalr, ranker=ranker, metric=metric, refiners_list=['bt_nllb', 'bt_bing'], overlap=True, agg=True) for ranker, metric in product(settings['ranker'], settings['metric'])]
        # for ranker, metric in product(ranker_list, metric_list):
        #     files = get_plat_files(infile, ds, ranker, metric, False)
        #     merge_plat_files(infiles=files, output=f'{output}/rag.vs.original/merge/{ds}.{ranker}.{metric}.agg.platinum.merged.tsv', rm=f'{ranker}.{metric}')
        #     compare_refiners(infile=f'{output}/rag.vs.original/merge/{ds}.{ranker}.{metric}.agg.platinum.merged.tsv', output=f'{output}/rag.vs.original/compare/{ds}.{ranker}.{metric}.agg.platinum.compare.csv', globalr=globalr, ranker=ranker, metric=metric, overlap=False, agg=False)
        # get_predictions(infile=f'./output/supervised/test/{ds}/bm25.map.{selected_refiner}', output=f'./output/{ds}/super.{selected_refiner}', original=f'./output/{ds}/original')
        # [get_agg(infile, output, ranker, metric) for ranker, metric in product(settings['ranker'], settings['metric'])]

    # get_avg(ranker_list, metric_list, ds_list, output)
    # combine_refiner_results(infile=f'{output}', output=f'{output}/all.platinum.compare.tsv', datasets=['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b'], ranker_metrics=product(['qld', 'bm25'], ['map', 'ndcg', 'recip_rank']))
    # combine_refiner_results(infile=f'{output}/rag.vs.original/compare/', output=f'{output}/rag.vs.original/compare.rag.all.datasets', datasets=ds_list, ranker_metrics=product(ranker_list, metric_list))
    # create_analyze_table('pred.')
    # analyze_similarity(infile=f'./output/', output=f'./output/analyze')
    # create_analyze_table('bt_bing')
    # refiner_distribution_table(infile='./output', output='./output/analyze/chart')
    # count_query_len()
    get_k_parameters(infile=f'./output', output='./output')
