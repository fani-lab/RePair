import pandas as pd
import numpy as np


# example of how to send a file path to the class
# file_path = '../output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.recip_rank.10.agg.gold.tsv'
class get_stats():
    def __init__(self, datasets, file_path):
        self.file_path = file_path
        self.datasets = datasets
        self.map_ds = pd.read_csv(file_path, sep='\t', encoding='utf-8', names=['qid', 'order', 'query', 'bm25'])
        self.num_rows, self.column_rows = self.map_ds.shape

    #  # of original queries who has a refined query with score 1.0
    def count_original_queries_with_perfect_refinement(self):
        original_queries_with_perfect_refinement = 0
        row_count = self.num_rows - 1
        i = 1
        while i <= row_count:
            if self.map_ds['order'][i] == '-1':
                if i < row_count and self.map_ds['order'][i + 1] != '-1':
                    print(self.map_ds['bm25'][i + 1])
                    if float(self.map_ds['bm25'][i + 1]) == 1.0:
                        original_queries_with_perfect_refinement += 1
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return original_queries_with_perfect_refinement

    # of original queries with no refined query
    def count_original_queries_without_refinement(self):
        row_count = self.num_rows - 1
        i = 1
        original_queries_without_refinement = 0
        while i <= row_count:
            if self.map_ds['order'][i] == '-1':
                if i < row_count and self.map_ds['order'][i + 1] == '-1':
                    original_queries_without_refinement += 1
                    i += 1
                else:
                    if i == row_count and self.map_ds['order'][i] == '-1':
                        original_queries_without_refinement += 1
                        i += 1
                    i += 1
            else:
                i += 1
        return original_queries_without_refinement

    # Max, Min, Avg of the # refined queries per original query
    def refined_queries_per_original_stats(self):
        row_count = self.num_rows - 1
        i = 1
        refined_queries_per_original = []
        while i < row_count:
            if self.map_ds['order'][i] == '-1' and self.map_ds['order'][i + 1] != '-1':
                num_refined = 0
                while i < row_count and self.map_ds['order'][i + 1] != '-1':
                    num_refined += 1
                    i += 1
                refined_queries_per_original.append(num_refined)
            else:
                i += 1
        print(refined_queries_per_original)
        return np.max(refined_queries_per_original), np.min(refined_queries_per_original), np.mean(
            refined_queries_per_original)

    # of original queries and refined queries
    def count_total_queries(self):
        original_queries = 0
        refuned_queries = 0
        for values in self.map_ds.loc[1:, 'order']:
            if values == '-1':
                original_queries += 1
            elif values != '-1':
                refuned_queries += 1
        total_queries = [original_queries, refuned_queries]
        return total_queries

    def combined_stats(self):
        i = 1
        row_count = self.num_rows - 1

        # delta_lengths_stats
        delta_lengths = []

        # delta_scores_stats
        delta_scores = []

        # original_queries_score_stats
        original_queries_score = []

        # original_queries_length_stats
        original_queries_length = []

        # get_num_refined_original_queries
        num_refined_original_queries = 0

        while i < row_count:
            if self.map_ds['order'][i] == '-1':
                if i < row_count and self.map_ds['order'][i + 1] != '-1':
                    # delta_scores_stats
                    original_score = float(self.map_ds['bm25'][i])
                    best_refined_score = float(self.map_ds['bm25'][i + 1])

                    delta_scores.append(best_refined_score - original_score)
                    # delta_lengths_stats
                    original_length = len(self.map_ds['query'][i])
                    best_refined_length = len(self.map_ds['query'][i + 1])

                    delta_lengths.append(best_refined_length - original_length)

                    # original_queries_score_stats
                    original_queries_score.append(float(self.map_ds['bm25'][i]))

                    # original_queries_length_stats
                    original_queries_length.append(len(self.map_ds['query'][i]))

                    # get_num_refined_original_queries
                    num_refined_original_queries += 1
                    i += 1
                else:
                    i += 1
            else:
                i += 1

        return np.max(delta_lengths), np.min(delta_lengths), np.mean(delta_lengths), np.max(delta_scores), np.min(
            delta_scores), np.mean(delta_scores), np.max(original_queries_score), np.min(
            original_queries_score), np.mean(original_queries_score), np.max(original_queries_length), np.min(
            original_queries_length), np.mean(original_queries_length), num_refined_original_queries

    # stuff I combined
    # # Max, Min, Avg of delta scores between original query and the best refined query
    # # Max, Min, Avg of delta of lengths between original query and the best refined query
    # # Max, Min, Avg of the scores for the original queries
    # # Max, Min, Avg of length for the original queries
    # # number of original queries that have a refined query
