import pandas as pd
import csv
def create_toy_dataset(dataset):
    dataframe = next(dataset)
    print(dataframe)
    dataframe.to_csv(f"../Data/{dataset.handles.handle.name.split('/')[-3]}/toy/{dataset.handles.handle.name.split('/')[-1]}",sep="\t",index = False)