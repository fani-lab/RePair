import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
datasets = ['diamond', 'platinum', 'gold']
def plot_stats(box_path):
    for ds in datasets:
        map_ds = pd.read_csv(f'{box_path}/{ds}.tsv', sep='\t', encoding='utf-8', names=['qid', 'i', 'i_map', 't', 't_map'])
        map_ds.sort_values(by='i_map', inplace=True)
        stats = map_ds.groupby(np.arange(len(map_ds))//(len(map_ds)/10)).mean()
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

plot_stats('../../output/msmarco.passage/t5.base.gc.docs.query/bm25.map.datasets')