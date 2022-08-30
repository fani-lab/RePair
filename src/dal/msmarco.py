import json
import csv
import os
import json
import pandas as pd

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

def fetchDocument(docId,searcher):
    try:
        doc = searcher.doc(docId)
        json_doc = json.loads(doc.raw())
        return json_doc['contents']
    except:
        raise ValueError('the provided value is not found in the passage')

print(fetchDocument(2912794,searcher))
def msmarco(qrelsFile,queriesFile):
    qrels_source = pd.read_csv(qrelsFile, sep='\t')
    with open(queriesFile) as queries:
        query_source = csv.reader(queries, delimiter='\t')
        next(query_source)

        for line in query_source:
            fetch_qid = qrels_source[qrels_source['qid'] == int(line[0])]
            get_global_document = int(fetch_qid['pid'])
            store_document = fetchDocument(get_global_document, searcher)
            print(store_document)
