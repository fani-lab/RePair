import csv, json
import pandas as pd
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

def msmarco(input, output):
    queries_source = pd.read_csv(f'{input}/queries.target.tsv', sep="\t", index_col=False)
    with open(f'{input}/qrels.target.tsv') as f:
        qrels = csv.reader(f, delimiter='\t')
        next(qrels)
        with open(f'{output}/qrels.target.tsv', 'w') as pf, open(f'{input}/queries.target.tsv', 'w') as qf:
            pf.write("pid\tpassage\n")
            qf.write("qid\tquery\n")
            for line in pf:
                fetch_qid = queries_source.loc[queries_source['qid'] == int(line[0])]
                try:
                    doc = searcher.doc(line[2])#docid
                    json_doc = json.loads(doc.raw())
                    retrieved_passage = fix_text(json_doc['contents'])
                except:
                    raise ValueError('the provided docid is not valid')

                qf.write(f"{fetch_qid['qid'].squeeze()}\t{fetch_qid['query'].squeeze()}\n")
                qp.write(f'{line[2]}\t{retrieved_passage}\n')
