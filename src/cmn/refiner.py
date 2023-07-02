import pandas as pd

# creates a train_test_split using pandas
datasets = ['diamond', 'platinum', 'gold']


def train_test_split(input,train_split=0.8):
    for ds in datasets:
        refiner_ds = pd.read_csv(f'{input}/{ds}.tsv', sep='\t', encoding='utf-8', names=['qid', 'query', 'map', 'query_', 'map_'])
        train = refiner_ds.sample(frac=train_split, random_state=200)
        test = refiner_ds.drop(train.index)
        train.to_csv(f'{input}/{ds}.train.tsv', sep='\t', index=False, header=False, columns=['query', 'query_'])
        test.to_csv(f'{input}/{ds}.test.tsv', sep='\t', index=False, header=False, columns=['query', 'query_'])
        print(f'saving {ds} with {train_split * 100}% train split and  {int(1 - train_split) * 100}% test split at {input} ')