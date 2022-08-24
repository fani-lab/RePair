from os.path import exists
from common.add_col_name import add_colname

def add_headers():
    if (exists('../Data/msmarco/qrels.train.tsv')!=True):
        return add_colname("../Raw/msmarco/qrels.train.tsv",["qid","did","pid","relevancy"])
check_to_add_headers = add_headers()

