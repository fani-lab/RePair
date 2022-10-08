import pandas as pd
from ftfy import fix_text
def fix_context_encoding(file_location):
    qrels_file = file_location
    df = pd.read_csv(qrels_file,sep='\t')
    df["passage"] = df["passage"].apply(lambda x: x.replace(x, fix_text(x)))
    df.to_csv(qrels_file,sep= "\t",index = False)