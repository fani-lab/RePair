from genericpath import exists
import pandas as pd

def add_colname(filename,colname_as_list,datasetName):
    """
    function inputs a filename,column names and dataset name to add headers.
    This is usually required once for adding headers to the data files from the raw folder.
    since the qrels do no have column names as they exist.
    """
    store = f'../Raw/{datasetName}/'
    clean_data_location = f'../Data/{datasetName}/'
    file_location = clean_data_location+filename
    file = pd.read_csv(store+filename,sep='\t')   
    file.columns = list(colname_as_list)
    file_location = file.to_csv(file_location, sep="\t", index=False)
    print(f' printing dataframe after adding headers:\n {file.head()}')
    return file_location

