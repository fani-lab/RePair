import json
import os
from os.path import isfile, join
import pandas as pd
import ir_datasets
from tqdm import tqdm
dataset = ir_datasets.load("aol-ia")
def initiate_queries_qrels(input):
    """
    :param input: location to store the files
    :return: creates a qrels and query file from a dataframe
    """
    queries = {'id': list(), 'query': list()}
    qrels = {'qid': list(), 'did': list(), 'iter': list(), 'rel': list()}
    #loop to create qrels file
    if not (isfile(join(input, 'qrels.tsv'))):
        print(f'creating qrels file in {input}')
        for qrel in tqdm(dataset.qrels_iter(), total=19442629):
            qrels['qid'].append(qrel.query_id)
            qrels['did'].append(qrel.doc_id)
            qrels['iter'].append(qrel.iteration)
            qrels['rel'].append(qrel.relevance)
        qrels_df = pd.DataFrame.from_dict(qrels)
        qrels_df.to_csv(f'{input}/qrels.tsv', sep='\t', encoding='UTF-8', index=False, header=1)
        print('qrels file is ready for use')
    if not (isfile(join(input, 'queries.tsv'))):
        print(f'creating queries file in {input}')
        for query in tqdm(dataset.queries_iter(), total=9966939):
            queries['id'].append(query.query_id)
            queries['query'].append(query.text)
        queries_df = pd.DataFrame.from_dict(queries)
        queries_df.dropna()
        queries_df.to_csv(f'{input}/queries.tsv', sep='\t', encoding='UTF-8', index=False, header=1)
        print('queries file is ready for use')

def create_json_collection(input,index_item):
    """
    logic for this code was taken from https://github.com/castorini/anserini-tools/blob/7b84f773225b5973b4533dfa0aa18653409a6146/scripts/msmarco/convert_collection_to_jsonl.py
    :param input: folder name to create docs
    :return: collection of jsonl
    """
    if not os.path.isdir(os.path.join(input, index_item)): os.makedirs(os.path.join(input, index_item))
    if not isfile(join(input, index_item, 'docs11.json')):
        max_docs_per_file = 1000000
        file_index = 0
        print('Converting aol docs into jsonl collection...')
        for i, doc in enumerate(dataset.docs_iter()):
            doc_id, doc_text = doc.doc_id, doc.index_item
            if i % max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = join(input, index_item, 'docs{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                file_index += 1
            output_dict = {'id': doc_id, 'contents': doc_text}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')
            if i % 100000 == 0:
                print(f'Converted {i:,} docs, writing into file {file_index}')
    print('completed writing to file!')


