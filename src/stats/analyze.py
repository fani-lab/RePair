from src.refinement.refiner_param import refiners
from src.refinement.lang_code import other, nllb
import matplotlib.pyplot as plt
from src.param import settings
from itertools import product
import pandas as pd
import json
import sys
import os
import re

sys.path.extend(["./src/output"])


def get_refiner_list(category):
    names = list(refiners[category].keys())
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
        df_list = [pd.read_csv(f'{infile}.{ds}.{r}.{m}.csv', skiprows=1, names=['category', 'refiner', f'{ds}_|q**|', f'{ds}_%']) for ds in datasets]
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
    if len(refiners_list) == 0: refiners_list = get_refiner_list('global') + get_refiner_list('local') + ['-1']
    if agg:
        # df['order'] = df['order'].apply(lambda x: 'bt' if 'bt' in x else x)
        # bt_rows = df[df['order'] == 'bt']
        # max_indices = bt_rows.groupby(['qid', 'order'])[f'{ranker}.{metric}'].idxmax()
        # df_bt = bt_rows.loc[max_indices]
        # df = pd.concat([df_bt, df[df['order'] != 'bt']], ignore_index=True)
        filtered_dfs = []
        for refiner_name in refiners_list:
            filtered_df = df.loc[df[df['order'].str.contains(refiner_name)].groupby('qid')[f'{ranker}.{metric}'].idxmax()]
            filtered_df['order'] = refiner_name
            filtered_dfs.append(filtered_df)
        df = pd.concat(filtered_dfs, ignore_index=True)
        df.reset_index(drop=True, inplace=True)

    # df = df[df['order'].str.contains('|'.join(refiners_list))]
    # Selecting q** (The best query among all refined queries for each qid)
    if not overlap:
        max_indices = df.groupby('qid')[f'{ranker}.{metric}'].idxmax()
        df = df.loc[max_indices]

    df.reset_index(drop=True, inplace=True)
    # plot_chart(df, ranker, metric, f'{output}.png')
    df['order'] = df['order'].apply(lambda x: 'q^' if '-1' in x else x)

    # Write results in a file
    final_table = pd.DataFrame(columns=['category', 'refiner', '|q**|', '%'])
    for refiner, group in df.groupby('order'):
        c = 'global 'if any(name in refiner for name in globalr) else 'local'
        final_table.loc[len(final_table)] = [c, refiner, len(group), (len(group)/num_q)*100]
    final_table = final_table.sort_values(by=['category', 'refiner'], ascending=[True, True])
    final_table.to_csv(f'{output}.csv', index=False)

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


def rename_files_with_nllb(folder_path, ranker='', metric=''):
    def get_key_from_value(value):
        for key, val in nllb.items():
            if val.lower() == value:
                return key
        return None

        # Get list of files in the folder
    files = os.listdir(folder_path)

    # Filter files that contain 'bt' but not 'bing'
    filtered_files = [file for file in files if 'nllb' in file]

    # Rename files
    for file in filtered_files:
        # Split the filename into parts
        parts = file.split('nllb_')
        if len(parts) == 2:
            dots = parts[1].split('.')
            lang = get_key_from_value(dots[0])
            # Construct the new filename with 'nllb' added between 'bt_' and the rest of the filename
            if lang is None: continue
            if ranker == '' and metric == '': new_name = f"{parts[0]}nllb_{lang}"
            else: new_name = f"{parts[0]}nllb_{lang}.{('.').join(dots[1:])}"
            # Rename the file
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
            print(f"Renamed {file} to {new_name}")


def get_predictions(infile, output, original):
    for baseline in ['acg', 'seq2seq', 'hredqs']:
        prediction = pd.read_json(f'{infile}/{baseline}/{baseline}.e100.json')
        reference = pd.DataFrame(original, names=['qid', 'previous_queries', ])


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


if __name__ == '__main__':
    globalr = get_refiner_list('global')
    localr = get_refiner_list('local')
    selected_refiner = 'nllb'
    # for ds in ['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b']:
    # #     infile = f'./output/{ds}'
    #     output = f'./output/analyze/supervised'
    #     if not os.path.isdir(output): os.makedirs(output)
        # rename_files_with_nllb(f'{infile}')
        # [rename_files_with_nllb(f'{infile}/{ranker}.{metric}', ranker, metric) for ranker, metric in product(settings['ranker'], settings['metric'])]
        # [compare_refiners(infile=f'{infile}/{ranker}.{metric}/{ranker}.{metric}.agg.platinum.tsv', output=f'{output}/compare.bt.refiners.{ds}.{ranker}.{metric}', globalr=globalr, ranker=ranker, metric=metric, refiners_list=['bt_nllb', 'bt_bing'], overlap=False, agg=True) for ranker, metric in product(settings['ranker'], settings['metric'])]
        # [compare_refiners(infile=f'{infile}/{ranker}.{metric}/{ranker}.{metric}.agg.rag.platinum.tsv', output=f'{output}/compare.refiners.rag.{ds}.{ranker}.{metric}', globalr=globalr, ranker=ranker, metric=metric, overlap=False, agg=True) for ranker, metric in product(settings['ranker'], settings['metric'])]
        # [get_predictions(infile=f'./output/supervised/{ds}/{ranker}.{metric}.{selected_refiner}', output=f'{output}/{ds}/super.{ranker}.{metric}.{selected_refiner}', original=f'./output/{ds}/refiner.original') for ranker, metric in product(settings['ranker'], settings['metric'])]

    # combine_refiner_results(infile='./output/analyze/compare.bt.refiners', output='./output/analyze/compare.bt.refiners.all.datasets', datasets=['dbpedia', 'robust04', 'antique', 'gov2', 'clueweb09b'], ranker_metrics=product(settings['ranker'], settings['metric']))
    # combine_refiner_results(infile='./output/analyze/compare.refiners.rag', output='./output/analyze/compare.refiners.rag.all.datasets', datasets=['robust04', 'antique', 'dbpedia'], ranker_metrics=product(settings['ranker'], settings['metric']))
    # create_analyze_table('pred.')
    analyze_similarity(infile=f'./output/', output=f'./output/analyze')
    # create_analyze_table('bt_bing')
    # refiner_distribution_table(infile='./output', output='./output/analyze/chart')
    # count_query_len()
