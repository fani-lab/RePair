from os.path import exists
from common.add_col_name import add_colname


raw_data_location  = './Raw/msmarco/'
clean_data_location = './Data/msmarco/'
def add_headers(filename,colname):

    if (exists(clean_data_location+filename)!=True):
        return add_colname(raw_data_location+filename,colname)
    else:
        print(f'..fetching queries from {filename}')

headers_for_msmarco_queries = add_headers('qrels.train.tsv',["qid","did","pid","relevancy"])
headers_for_msmarco_qrels = add_headers('queries.train.tsv',["qid","query"])