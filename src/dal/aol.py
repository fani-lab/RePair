import json
import os
from os.path import isfile, join
import pandas as pd
import ir_datasets
from tqdm import tqdm

dataset = ir_datasets.load("aol-ia")


def fetch_content(index_item, doc):
    if index_item == 'title':
        return doc.title
    elif index_item == 'text':
        return doc.text
    else:
        return ' '.join([doc.title, doc.text])


def initiate_queries_qrels(input):
    """
    :param input: location to store the files
    :return: creates a duplicate and no duplicate qrels and query file  from a dataframe
    """


    # loop to create qrels file format - qid iter did rel
    if not (isfile(join(input, 'qrels.tsv'))):
        qrels = {'qid': list(), 'iter': list(), 'did': list(), 'rel': list()}
        print(f'creating qrels file in {input}')
        for qrel in tqdm(dataset.qrels_iter(), total=19442629):
            qrels['qid'].append(qrel.query_id)
            qrels['did'].append(qrel.doc_id)
            qrels['iter'].append(qrel.iteration)
            qrels['rel'].append(qrel.relevance)
        qrels_df = pd.DataFrame.from_dict(qrels)
        qrels_df.to_csv(f'{input}/qrels.tsv', sep='\t', encoding='UTF-8', index=False, header=False)
        qrels_df.drop_duplicates(inplace=True)
        qrels_df.to_csv(f'{input}/qrels.nodups.tsv', sep='\t', encoding='UTF-8', index=False, header=False)
        qrels_df.drop_duplicates(subset=['qid', 'pid'], inplace=True)
        qrels_df.to_csv(f'{input}/qrels.nodupsiter.tsv', sep='\t', encoding='UTF-8', index=False, header=False)
        print('qrels file is ready for use')
    if not (isfile(join(input, 'queries.tsv'))):
        queries = {'id': list(), 'query': list()}
        print(f'creating queries file in {input}')
        for query in tqdm(dataset.queries_iter(), total=9966939):
            queries['id'].append(query.query_id)
            queries['query'].append(query.text)
        queries_df = pd.DataFrame.from_dict(queries)
        queries_df.to_csv(f'{input}/queries.tsv', sep='\t', encoding='UTF-8', index=False, header=False)
        queries_df.dropna(inplace=True)
        queries_df.drop_duplicates(inplace=True)
        queries_df.to_csv(f'{input}/queries.nodups.tsv', sep='\t', encoding='UTF-8', index=False, header=False)
        toy_sample = queries_df.sample(n=500)
        toy_sample.to_csv(f'{input.replace("aol","toy.aol")}/queries.nodups.tsv', sep='\t', encoding='UTF-8', index=False, header=False)
        print('queries file is ready for use')

def create_json_collection(input, index_item='title_and_text'):
    """
    logic for this code was taken from https://github.com/castorini/anserini-tools/blob/7b84f773225b5973b4533dfa0aa18653409a6146/scripts/msmarco/convert_collection_to_jsonl.py
    :param input: folder name to create docs
    :return: collection of jsonl
    """
    if not os.path.isdir(os.path.join(input, index_item)): os.makedirs(os.path.join(input, index_item))
    if not isfile(join(input, index_item, 'docs00.json')):
        max_docs_per_file = 1000000
        file_index = 0
        print('Converting aol docs into jsonl collection...')
        for i, doc in enumerate(dataset.docs_iter()):  # doc returns doc_id, title, text, url, ia_url
            doc_id, doc_content = doc.doc_id, fetch_content(index_item,doc)
            if i % max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = join(input, index_item, 'docs{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                file_index += 1
            output_dict = {'id': doc_id, 'contents': doc_content}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')
            if i % 100000 == 0:
                print(f'Converted {i:,} docs, writing into file {file_index}')
    print('completed writing to file!')


# def to_pair(input, output, cat=True):
#     queries = pd.read_csv(f'{input}/queries.nodups.tsv', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
#     qrels = pd.read_csv(f'{input}/qrels.nodupsiter.tsv', sep='\t', index_col=False, names=['qid', 'iter', 'did', 'relevancy'], header=None)
#     queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
#     doccol = 'docs' if cat else 'doc'
#     queries_qrels[doccol] = queries_qrels['pid'].progress_apply(to_txt) #100%|██████████| 532761/532761 [00:32<00:00, 16448.77it/s]
#     queries_qrels['ctx'] = ''
#     if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False).agg({'did': list, 'pid': list, doccol: ' '.join})
#     queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
#     #removes duplicates from qrels file
#     if not isfile(join(input, 'qrels.train.nodups.tsv')):
#         qrels.drop_duplicates(inplace=True)
#         qrels.to_csv(f'{input}/qrels.train.nodups.tsv', sep='\t', index=False, header=None)
#     return queries_qrels