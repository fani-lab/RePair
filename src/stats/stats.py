import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stats.get_stats import get_stats

datasets = ['diamond', 'platinum', 'gold']


def plot_stats(box_path):
    for ds in datasets:
        map_ds = pd.read_csv(f'{box_path}/{ds}.tsv', sep='\t', encoding='utf-8', names=['qid', 'i', 'i_map', 't', 't_map'])
        map_ds.sort_values(by='i_map', inplace=True)
        stats = map_ds.groupby(np.arange(len(map_ds)) // (len(map_ds) / 10)).mean(numeric_only=True)
        X = [x for x in range(1, 11)]
        original_mean = stats['i_map']
        changes_mean = stats['t_map']
        X_axis = np.arange(len(X))
        plt.bar(X_axis - 0.2, original_mean, 0.4, label='original')
        plt.bar(X_axis + 0.2, changes_mean, 0.4, label='change')
        plt.xticks(X_axis, labels=range(1, 11))
        plt.xlabel("buckets")
        plt.ylabel("MAP")
        plt.title(f'{ds} dataset for {box_path.split("/")[3]}')
        print(f'saving stats image for {ds} at {box_path}')
        plt.savefig(f'{box_path}/{ds}.jpg')
        plt.clf()


plot_stats('../output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.map.boxes')
file_path = '../output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.recip_rank.10.agg.gold.tsv'

s = get_stats(datasets, file_path)


stats = {}
# stats['original_queries'] = s.count_original_queries()
# stats['refined_queries'] = s.count_refined_queries()
stats['count_total_queries'] = (lambda x: {'original': x[0], 'refined': x[1]})(s.count_total_queries())
stats['count_original_queries_with_perfect_refinement'] = s.count_original_queries_with_perfect_refinement()
stats['count_original_queries_without_refinement'] = s.count_original_queries_without_refinement()
stats['refined_queries_per_original_stats'] = s.refined_queries_per_original_stats()
stats['delta_lengths_stats'] = (lambda x: {'max': x[0], 'min': x[1], 'mean': x[2]})(s.combined_stats())
stats['delta_scores_stats'] = (lambda x: {'max': x[3], 'min': x[4], 'mean': x[5]})(s.combined_stats())
stats['original_queries_score_stats'] = (lambda x: {'max': x[6], 'min': x[7], 'mean': x[8]})(s.combined_stats())
stats['original_queries_length_stats'] = (lambda x: {'max': x[9], 'min': x[10], 'mean': x[11]})(s.combined_stats())
stats['get_num_refined_original_queries'] = s.combined_stats()[12]


print(f'count_original_queries and refined queries: {stats["count_total_queries"]}')
print(f'get_num_refined_original_queries: {stats["get_num_refined_original_queries"]}')
print(f'count_original_queries_with_perfect_refinement: {stats["count_original_queries_with_perfect_refinement"]}')
print(f'count_original_queries_without_refinement: {stats["count_original_queries_without_refinement"]}')
print(f'refined_queries_per_original_stats: {stats["refined_queries_per_original_stats"]}')
print(f'original_queries_length_stats: {stats["original_queries_length_stats"]}')
print(f'original_queries_score_stats: {stats["original_queries_score_stats"]}')
print(f'delta_lengths_stats: {stats["delta_lengths_stats"]}')
print(f'delta_scores_stats: {stats["delta_scores_stats"]}')
