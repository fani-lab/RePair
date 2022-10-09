import csv, json
import pandas as pd
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

def msmarco(input, output):

    # https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    if not searcher:
        # sometimes you need to manually download the index ==> https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-manually-download-indexes
        searcher = LuceneSearcher('./../data/raw/msmarco/lucene-index.msmarco-v1-passage.20220131.9ea315/')
        if not searcher: raise ValueError("Lucene searcher cannot find/build msmarco index!")

    queries = pd.read_csv(f'{input}/queries.train.tsv', sep="\t", index_col=False, names=['qid', 'query'], header=None)
    qrels = pd.read_csv(f'{input}/qrels.train.tsv', sep="\t", index_col=False, names=['qid', 'did', 'pid', 'relevancy'], header=None)
    with open(f'{output}/qrels.train.tsv', 'w', encoding='utf-8') as pf, open(f'{output}/queries.train.tsv', 'w', encoding='utf-8') as qf:
        pf.write("pid\tpassage\n")
        qf.write("qid\tquery\n")
        for row in tqdm(qrels.itertuples(), total=qrels.shape[0]):#100%|██████████| 532761/532761 [10:24<00:00, 853.57it/s]
            fetch_qid = queries.loc[queries['qid'] == row.qid]
            try:
                # The``docid`` is overloaded: if it is of type ``str``, it is treated as an external collection ``docid``;
                # if it is of type ``int``, it is treated as an internal Lucene``docid``.
                # stupid!!
                doc = searcher.doc(str(row.pid))#passage id
                json_doc = json.loads(doc).raw()
                retrieved_passage = fix_text(json_doc['contents'])
            except Exception as e:
                raise e

            qf.write(f"{fetch_qid['qid'].squeeze()}\t{fetch_qid['query'].squeeze()}\n")
            pf.write(f"{row.pid}\t{retrieved_passage}\n")
