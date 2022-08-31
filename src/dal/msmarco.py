import csv
import json
import pandas as pd

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

def fetchDocument(doc_id,searcher):
    try:
        doc = searcher.doc(doc_id)
        json_doc = json.loads(doc.raw())
        return json_doc['contents']
    except:
        raise ValueError('the provided value is not found in the passage')

def msmarco(qrels_file,queries_file):
    qrels_source = pd.read_csv(qrels_file, sep='\t', index_col=False)
    with open(queries_file) as queries:
        query_source = csv.reader(queries, delimiter='\t')
        next(query_source)
        for line in query_source:
            fetch_qid = qrels_source[qrels_source['qid'] == int(line[0])]
            get_global_document = int(fetch_qid['pid'])
            store_document = fetchDocument(get_global_document, searcher)