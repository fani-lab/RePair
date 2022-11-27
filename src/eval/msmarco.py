import os
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


def getHits(input, output):
    predicted_file_list = os.listdir(input)
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    if not searcher:
        searcher = LuceneSearcher('./../data/raw/msmarco/lucene-index.msmarco-v1-passage.20220131.9ea315/')
        if not searcher: raise ValueError("Lucene searcher cannot find/build msmarco index!")
    #get hits using ranker.
    for iter in predicted_file_list:
        predicted_queries = pd.read_fwf(f'{input}/{predicted_file_list[0]}', header=None)
        with open(f'{output}/msmarcoPassageRun{iter.split(".")[0][-3:]}.txt', 'w', encoding='utf-8') as retrieved_passage:
            for row in tqdm(predicted_queries.itertuples(), total=predicted_queries.shape[0]):
                hits = searcher.search(row[1])
                for i in range(len(hits)):
                    retrieved_passage.write(f'{i+1}\t{hits[i].docid:7}\t{hits[i].score:.5f}\n')


