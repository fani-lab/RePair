from genericpath import exists
import pandas as pd

def add_colname(file_location,colname_as_list):
    data_location ='../Data/msmarco/qrels.train.tsv'  
    file = pd.read_csv(file_location,sep='\t')   
    file.columns = list(colname_as_list)
    file.to_csv(data_location, sep="\t", index=False)
    print(f' printing dataframe after adding headers:\n {file.head()}')
    return data_location

